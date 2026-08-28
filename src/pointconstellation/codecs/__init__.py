"""External codec adapters used by compression benchmarks."""

from pointconstellation.codecs.external import (
    ExternalCodecResult,
    ExternalCodecSpec,
    run_external_codec,
    run_external_codec_batch,
)
from pointconstellation.codecs.gpcc import (
    GpccResult,
    GpccStreamBreakdown,
    OfficialMetricResult,
    Tmc3RatePoint,
    parse_gpcc_stream,
    read_ascii_ply,
    run_pc_error,
    run_tmc3,
    write_ascii_ply,
)

__all__ = [
    "ExternalCodecResult",
    "ExternalCodecSpec",
    "GpccResult",
    "GpccStreamBreakdown",
    "OfficialMetricResult",
    "Tmc3RatePoint",
    "parse_gpcc_stream",
    "read_ascii_ply",
    "run_external_codec",
    "run_external_codec_batch",
    "run_pc_error",
    "run_tmc3",
    "write_ascii_ply",
]
