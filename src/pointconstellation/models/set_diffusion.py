"""Conditional diffusion over unordered coordinate constellations.

The denoiser is intentionally slot-free: without positional embeddings, input-cloud
permutations leave its output unchanged and particle permutations permute its output.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pointconstellation.losses import pairwise_squared
from pointconstellation.quantization import quantize_coordinates


def farthest_point_constellation(points: Tensor, constellation_size: int) -> Tensor:
    """Select deterministic farthest-point anchors from a batched point set."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (batch, N, 3)")
    if constellation_size < 2 or constellation_size > points.shape[1]:
        raise ValueError("constellation_size must be between 2 and N")
    batch_size, num_points, _ = points.shape
    rows = torch.arange(batch_size, device=points.device)
    centroid = points.mean(dim=1, keepdim=True)
    farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    minimum_distances = torch.full(
        (batch_size, num_points),
        torch.inf,
        dtype=points.dtype,
        device=points.device,
    )
    selected = torch.empty(
        (batch_size, constellation_size), dtype=torch.long, device=points.device
    )
    for index in range(constellation_size):
        selected[:, index] = farthest
        anchor = points[rows, farthest][:, None, :]
        minimum_distances = torch.minimum(
            minimum_distances, ((points - anchor) ** 2).sum(dim=-1)
        )
        farthest = minimum_distances.argmax(dim=1)
    return points.gather(1, selected[:, :, None].expand(-1, -1, 3))


def _sinusoidal_embedding(timesteps: Tensor, width: int) -> Tensor:
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < width:
        embedding = torch.nn.functional.pad(embedding, (0, width - embedding.shape[1]))
    return embedding


class ConditionalSetDenoiser(nn.Module):
    """Predict particle noise conditioned on an unordered, variable-size cloud."""

    def __init__(
        self,
        *,
        max_constellation_size: int,
        feature_width: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if max_constellation_size < 2:
            raise ValueError("max_constellation_size must be at least 2")
        if feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.max_constellation_size = max_constellation_size
        self.feature_width = feature_width
        self.point_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        point_layer = nn.TransformerEncoderLayer(
            d_model=feature_width,
            nhead=num_heads,
            dim_feedforward=2 * feature_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.point_relations = nn.TransformerEncoder(
            point_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.particle_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(feature_width + 1, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.cross_attention = nn.ModuleList()
        self.particle_relations = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attention.append(
                nn.MultiheadAttention(
                    feature_width, num_heads, dropout=0.0, batch_first=True
                )
            )
            particle_layer = nn.TransformerEncoderLayer(
                d_model=feature_width,
                nhead=num_heads,
                dim_feedforward=2 * feature_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.particle_relations.append(particle_layer)
            self.norms.append(nn.LayerNorm(feature_width))
        self.output = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
        )

    def forward(
        self, noisy_constellation: Tensor, points: Tensor, timesteps: Tensor
    ) -> Tensor:
        if noisy_constellation.ndim != 3 or noisy_constellation.shape[-1] != 3:
            raise ValueError("noisy_constellation must have shape (batch, K, 3)")
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        if len(noisy_constellation) != len(points):
            raise ValueError("constellation and points batch sizes must match")
        if noisy_constellation.shape[1] > self.max_constellation_size:
            raise ValueError("constellation exceeds the configured maximum")
        if timesteps.shape != (len(points),):
            raise ValueError("timesteps must have shape (batch,)")

        context = self.point_relations(self.point_embedding(points))
        particles = self.particle_embedding(noisy_constellation)
        cardinality = noisy_constellation.new_full(
            (len(points), 1),
            noisy_constellation.shape[1] / self.max_constellation_size,
        )
        condition = self.condition_embedding(
            torch.cat(
                (
                    _sinusoidal_embedding(timesteps, self.feature_width).to(
                        noisy_constellation.dtype
                    ),
                    cardinality,
                ),
                dim=1,
            )
        )
        particles = particles + condition[:, None, :]
        for attention, relation, norm in zip(
            self.cross_attention,
            self.particle_relations,
            self.norms,
            strict=True,
        ):
            attended, _ = attention(particles, context, context, need_weights=False)
            particles = relation(norm(particles + attended))
        return self.output(particles)


class DiffusionSchedule(nn.Module):
    """A compact DDPM schedule with target-like forward and reverse processes."""

    def __init__(
        self,
        num_steps: int = 32,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
        return values[timesteps].reshape(len(target), 1, 1).to(target.dtype)

    def q_sample(
        self,
        clean: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply forward noising and return both noisy coordinates and noise."""

        if timesteps.shape != (len(clean),):
            raise ValueError("timesteps must have shape (batch,)")
        if (timesteps < 0).any() or (timesteps >= self.num_steps).any():
            raise ValueError("timestep is outside the diffusion schedule")
        sampled_noise = torch.randn_like(clean) if noise is None else noise
        if sampled_noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        alpha_bar = self._extract(self.alpha_bars, timesteps, clean)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * sampled_noise
        return noisy, sampled_noise

    def p_step(
        self,
        denoiser: ConditionalSetDenoiser,
        current: Tensor,
        points: Tensor,
        timestep: int,
        *,
        stochastic: bool,
    ) -> Tensor:
        batch_timesteps = torch.full(
            (len(current),), timestep, dtype=torch.long, device=current.device
        )
        predicted_noise = denoiser(current, points, batch_timesteps)
        alpha = self.alphas[timestep].to(current.dtype)
        alpha_bar = self.alpha_bars[timestep].to(current.dtype)
        mean = current - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()
        mean = mean / alpha.sqrt()
        if timestep and stochastic:
            variance = self.betas[timestep].to(current.dtype).sqrt()
            mean = mean + variance * torch.randn_like(current)
        return mean.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_from_fps(
        self,
        denoiser: ConditionalSetDenoiser,
        points: Tensor,
        constellation_size: int,
        *,
        bits: int,
        start_step: int | None = None,
        stochastic: bool = True,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Refine a noised FPS constellation and exactly quantize the result."""

        step = self.num_steps - 1 if start_step is None else start_step
        if step < 0 or step >= self.num_steps:
            raise ValueError("start_step is outside the diffusion schedule")
        initial = farthest_point_constellation(points, constellation_size)
        timesteps = torch.full(
            (len(points),), step, dtype=torch.long, device=points.device
        )
        current, _ = self.q_sample(initial, timesteps, noise=initial_noise)
        for timestep in range(step, -1, -1):
            current = self.p_step(
                denoiser,
                current,
                points,
                timestep,
                stochastic=stochastic,
            )
        return quantize_coordinates(current, bits)


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


@torch.no_grad()
def select_best_decoded_candidate(
    decoder: nn.Module,
    candidates: Tensor,
    target: Tensor,
    *,
    num_output_points: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select each sample's lowest-Chamfer candidate through a frozen decoder."""

    if candidates.ndim != 4 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape (batch, candidates, K, 3)")
    if candidates.shape[0] != target.shape[0]:
        raise ValueError("candidate and target batch sizes must match")
    batch_size, candidate_count, constellation_size, _ = candidates.shape
    flat = candidates.reshape(batch_size * candidate_count, constellation_size, 3)
    decoded = decoder(flat, num_output_points=num_output_points)
    repeated_target = (
        target[:, None]
        .expand(-1, candidate_count, -1, -1)
        .reshape(batch_size * candidate_count, target.shape[1], 3)
    )
    scores = _chamfer_per_sample(decoded, repeated_target).reshape(
        batch_size, candidate_count
    )
    indices = scores.argmin(dim=1)
    rows = torch.arange(batch_size, device=candidates.device)
    return candidates[rows, indices], indices, scores


def _hungarian_assignment(cost: list[list[float]]) -> list[int]:
    """Return an O(K^3) minimum-cost assignment for a square cost matrix."""

    size = len(cost)
    if not size or any(len(row) != size for row in cost):
        raise ValueError("cost matrix must be nonempty and square")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matching[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate_column - 1]
                    - u[current_row]
                    - v[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[matching[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    return assignment


def matched_set_rmse(first: Tensor, second: Tensor) -> float:
    """Compute permutation-invariant RMS separation under optimal matching."""

    if first.ndim != 2 or first.shape[-1] != 3 or first.shape != second.shape:
        raise ValueError("sets must have the same shape (K, 3)")
    squared = ((first[:, None] - second[None]) ** 2).sum(dim=-1).detach().cpu()
    cost = squared.tolist()
    assignment = _hungarian_assignment(cost)
    return math.sqrt(
        sum(cost[row][column] for row, column in enumerate(assignment)) / len(cost)
    )


def multimodality_gate(
    restart_constellations: Tensor,
    restart_distortions: Tensor,
    *,
    relative_distortion_tolerance: float = 0.03,
    min_matched_separation: float = 0.05,
    min_multimodal_fraction: float = 0.25,
) -> dict[str, object]:
    """Gate diffusion on separated restarts with comparable decoder distortion."""

    if restart_constellations.ndim == 3:
        restart_constellations = restart_constellations[None]
    if restart_distortions.ndim == 1:
        restart_distortions = restart_distortions[None]
    if restart_constellations.ndim != 4 or restart_constellations.shape[-1] != 3:
        raise ValueError("restart_constellations must have shape (B, R, K, 3)")
    if restart_distortions.shape != restart_constellations.shape[:2]:
        raise ValueError("restart_distortions must have shape (B, R)")
    if restart_constellations.shape[1] < 2:
        raise ValueError("at least two restarts are required")
    if relative_distortion_tolerance < 0 or min_matched_separation < 0:
        raise ValueError("gate tolerances cannot be negative")
    if not 0 <= min_multimodal_fraction <= 1:
        raise ValueError("min_multimodal_fraction must be between zero and one")

    sample_records: list[dict[str, object]] = []
    multimodal_count = 0
    for sample_constellations, sample_distortions in zip(
        restart_constellations, restart_distortions, strict=True
    ):
        best = float(sample_distortions.min().item())
        threshold = best * (1.0 + relative_distortion_tolerance) + 1e-12
        comparable = torch.nonzero(sample_distortions <= threshold).flatten().tolist()
        maximum_separation = 0.0
        for offset, first_index in enumerate(comparable):
            for second_index in comparable[offset + 1 :]:
                maximum_separation = max(
                    maximum_separation,
                    matched_set_rmse(
                        sample_constellations[first_index],
                        sample_constellations[second_index],
                    ),
                )
        sample_passed = (
            len(comparable) >= 2 and maximum_separation >= min_matched_separation
        )
        multimodal_count += int(sample_passed)
        sample_records.append(
            {
                "best_distortion": best,
                "comparable_restart_count": len(comparable),
                "maximum_matched_set_separation": maximum_separation,
                "multimodal": sample_passed,
            }
        )
    fraction = multimodal_count / len(sample_records)
    passed = fraction >= min_multimodal_fraction
    return {
        "passed": passed,
        "reason": (
            None
            if passed
            else "optimized restarts did not show enough separated, comparable modes"
        ),
        "multimodal_sample_fraction": fraction,
        "min_multimodal_fraction": min_multimodal_fraction,
        "relative_distortion_tolerance": relative_distortion_tolerance,
        "min_matched_separation": min_matched_separation,
        "samples": sample_records,
    }
