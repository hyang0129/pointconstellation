"""External codec adapters used by compression benchmarks."""

from pointconstellation.codecs.draco import (
    DRACO_RELEASE,
    DRACO_RELEASE_COMMIT,
    DracoCodecSpec,
    DracoResult,
    run_draco,
)
from pointconstellation.codecs.external import (
    ExternalCodecResult,
    ExternalCodecSpec,
    run_external_codec,
    run_external_codec_batch,
)
from pointconstellation.codecs.gpcc import (
    GpccResult,
    OfficialMetricResult,
    Tmc3RatePoint,
    read_ascii_ply,
    run_pc_error,
    run_tmc3,
    write_ascii_ply,
)
from pointconstellation.codecs.pcgcv2 import (
    PCGCV2_PAYLOAD_SUFFIXES,
    PCGCV2_UPSTREAM_COMMIT,
    PCGCV2_UPSTREAM_URL,
    Pcgcv2HarnessConfig,
    Pcgcv2RatePoint,
    pack_pcgcv2_payloads,
    pcgcv2_diversity_summary,
    point_set_sha256,
    unpack_pcgcv2_payloads,
)

__all__ = [
    "DRACO_RELEASE",
    "DRACO_RELEASE_COMMIT",
    "DracoCodecSpec",
    "DracoResult",
    "ExternalCodecResult",
    "ExternalCodecSpec",
    "GpccResult",
    "OfficialMetricResult",
    "PCGCV2_PAYLOAD_SUFFIXES",
    "PCGCV2_UPSTREAM_COMMIT",
    "PCGCV2_UPSTREAM_URL",
    "Pcgcv2HarnessConfig",
    "Pcgcv2RatePoint",
    "Tmc3RatePoint",
    "read_ascii_ply",
    "pack_pcgcv2_payloads",
    "pcgcv2_diversity_summary",
    "point_set_sha256",
    "run_draco",
    "run_external_codec",
    "run_external_codec_batch",
    "run_pc_error",
    "run_tmc3",
    "unpack_pcgcv2_payloads",
    "write_ascii_ply",
]
