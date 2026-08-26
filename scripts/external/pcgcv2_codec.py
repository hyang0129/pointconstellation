#!/usr/bin/env python3
"""Run pinned PCGCv2 encode/decode as two complete black-box processes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.codecs.pcgcv2 import (  # noqa: E402
    PCGCV2_PAYLOAD_SUFFIXES,
    pack_pcgcv2_payloads,
    unpack_pcgcv2_payloads,
)


def _load_runtime(upstream: Path, checkpoint: Path):
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    import torch
    from coder import Coder
    from data_utils import load_sparse_tensor, scale_sparse_tensor, write_ply_ascii_geo
    from pcc_model import PCCModel

    if not torch.cuda.is_available():
        raise RuntimeError("PCGCv2 requires a CUDA device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    model = PCCModel().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return (
        torch,
        device,
        Coder,
        load_sparse_tensor,
        scale_sparse_tensor,
        write_ply_ascii_geo,
        model,
    )


def encode(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    stream_path = args.stream.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "pcgcv2_payload"
    (
        _,
        device,
        Coder,
        load_sparse_tensor,
        scale_sparse_tensor,
        _,
        model,
    ) = _load_runtime(args.upstream_dir.resolve(), args.checkpoint.resolve())
    source = load_sparse_tensor(str(input_path), device)
    codec_input = (
        scale_sparse_tensor(source, factor=args.scaling_factor)
        if args.scaling_factor != 1.0
        else source
    )
    Coder(model=model, filename=str(prefix)).encode(codec_input)
    pack_pcgcv2_payloads(prefix, stream_path)
    for suffix in PCGCV2_PAYLOAD_SUFFIXES:
        Path(str(prefix) + suffix).unlink()


def decode(args: argparse.Namespace) -> None:
    stream_path = args.stream.resolve()
    reconstruction_path = args.reconstruction.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "pcgcv2_payload"
    unpack_pcgcv2_payloads(stream_path, prefix)
    (
        _,
        _,
        Coder,
        _,
        scale_sparse_tensor,
        write_ply_ascii_geo,
        model,
    ) = _load_runtime(args.upstream_dir.resolve(), args.checkpoint.resolve())
    reconstruction = Coder(model=model, filename=str(prefix)).decode(rho=args.rho)
    if args.scaling_factor != 1.0:
        reconstruction = scale_sparse_tensor(
            reconstruction, factor=1.0 / args.scaling_factor
        )
    coordinates = reconstruction.C.detach().cpu().numpy()[:, 1:]
    levels = (1 << args.position_bits) - 1
    if len(coordinates) and (coordinates.min() < 0 or coordinates.max() > levels):
        raise RuntimeError("PCGCv2 reconstruction lies outside the declared grid")
    write_ply_ascii_geo(str(reconstruction_path), coordinates)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("encode", "decode"))
    result.add_argument("--upstream-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--position-bits", type=int, choices=(6, 7, 8), required=True)
    result.add_argument("--scaling-factor", type=float, required=True)
    result.add_argument("--rho", type=float, required=True)
    result.add_argument("--input", type=Path)
    result.add_argument("--stream", type=Path, required=True)
    result.add_argument("--reconstruction", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.scaling_factor <= 0 or args.rho <= 0:
        raise ValueError("scaling-factor and rho must be positive")
    if args.mode == "encode":
        if args.input is None:
            raise ValueError("encode requires --input")
        encode(args)
    else:
        if args.reconstruction is None:
            raise ValueError("decode requires --reconstruction")
        decode(args)


if __name__ == "__main__":
    main()
