"""Execute Python through an allocated Jupyter GPU kernel."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    status: str = "ok"
    outputs: list[dict[str, Any]] = field(default_factory=list)


def connection_settings(
    base_url: str | None = None,
    password: str | None = None,
) -> tuple[str, str]:
    """Resolve connection data without reading or storing a secrets file."""

    resolved_url = base_url or os.environ.get("POINTCONSTELLATION_JUPYTER_URL")
    if not resolved_url:
        raise RuntimeError(
            "set POINTCONSTELLATION_JUPYTER_URL or pass base_url explicitly"
        )
    resolved_password = password
    if resolved_password is None:
        resolved_password = os.environ.get("POINTCONSTELLATION_JUPYTER_PASSWORD", "")
    return resolved_url.rstrip("/"), resolved_password


def _dependencies() -> tuple[Any, Any]:
    try:
        import requests
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "EmpireAI Jupyter access requires: pip install -e '.[cluster]'"
        ) from exc
    return requests, websocket


class JupyterExecutor:
    """Authenticated Jupyter REST/WebSocket execution context."""

    def __init__(
        self,
        base_url: str | None = None,
        password: str | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        requests, websocket = _dependencies()
        self._websocket = websocket
        self.base_url, self.password = connection_settings(base_url, password)
        self.ws_base = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        self.kernel_name = kernel_name or os.environ.get(
            "POINTCONSTELLATION_JUPYTER_KERNEL", "python3"
        )
        self.session = requests.Session()
        self.cookie = ""
        self.xsrf = ""

    def login(self, timeout: float = 15.0) -> None:
        page = self.session.get(f"{self.base_url}/login", timeout=timeout)
        page.raise_for_status()
        xsrf = self.session.cookies.get("_xsrf", "")
        response = self.session.post(
            f"{self.base_url}/login",
            data={"password": self.password, "_xsrf": xsrf},
            allow_redirects=True,
            timeout=timeout,
        )
        response.raise_for_status()
        self.cookie = "; ".join(
            f"{key}={value}" for key, value in self.session.cookies.items()
        )
        self.xsrf = self.session.cookies.get("_xsrf", "")

    def start_kernel(self, timeout: float = 15.0) -> str:
        response = self.session.post(
            f"{self.base_url}/api/kernels",
            json={"name": self.kernel_name},
            headers={"X-XSRFToken": self.xsrf},
            timeout=timeout,
        )
        response.raise_for_status()
        return str(response.json()["id"])

    def stop_kernel(self, kernel_id: str, timeout: float = 15.0) -> None:
        self.session.delete(
            f"{self.base_url}/api/kernels/{kernel_id}",
            headers={"X-XSRFToken": self.xsrf},
            timeout=timeout,
        )

    def run(self, code: str, *, timeout: float = 60.0) -> ExecutionResult:
        if not self.cookie:
            self.login()
        kernel_id = self.start_kernel()
        socket = self._websocket.create_connection(
            f"{self.ws_base}/api/kernels/{kernel_id}/channels",
            header={"Cookie": self.cookie, "X-XSRFToken": self.xsrf},
            timeout=timeout,
        )
        try:
            return self._execute(socket, code, timeout)
        finally:
            socket.close()
            self.stop_kernel(kernel_id)

    def _execute(self, socket: Any, code: str, timeout: float) -> ExecutionResult:
        message_id = str(uuid.uuid4())
        socket.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": message_id,
                        "msg_type": "execute_request",
                        "username": "",
                        "session": str(uuid.uuid4()),
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                    "channel": "shell",
                }
            )
        )
        result = ExecutionResult()
        deadline = time.time() + timeout
        while time.time() < deadline:
            socket.settimeout(max(0.1, deadline - time.time()))
            try:
                message = json.loads(socket.recv())
            except self._websocket.WebSocketTimeoutException:
                result.status = "timeout"
                result.error = f"execution timed out after {timeout} seconds"
                break
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            message_type = message.get("msg_type", "")
            content = message.get("content", {})
            if message_type == "stream":
                destination = "stdout" if content.get("name") == "stdout" else "stderr"
                setattr(
                    result,
                    destination,
                    getattr(result, destination) + content.get("text", ""),
                )
            elif message_type in {"display_data", "execute_result"}:
                data = content.get("data", {})
                result.outputs.append(data)
                if text := data.get("text/plain", ""):
                    result.stdout += text + "\n"
            elif message_type == "error":
                result.status = "error"
                result.error = "\n".join(content.get("traceback", []))
            elif message_type == "execute_reply":
                if result.status != "error":
                    result.status = content.get("status", "ok")
                break
        return result

    def __enter__(self) -> JupyterExecutor:
        self.login()
        return self

    def __exit__(self, *_: object) -> None:
        self.session.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", nargs="?", help="Python source; stdin when omitted")
    parser.add_argument("--url")
    parser.add_argument("--password")
    parser.add_argument("--kernel")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    code = args.code if args.code is not None else sys.stdin.read()
    with JupyterExecutor(args.url, args.password, kernel_name=args.kernel) as executor:
        result = executor.run(code, timeout=args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.error:
        sys.stderr.write(result.error + "\n")
    raise SystemExit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    main()
