from __future__ import annotations

import pytest

from pointconstellation.rate_accounting import (
    AMORTIZATION_CORPUS_SIZES,
    amortized_bpp,
    amortized_bpp_table,
    model_amortization,
    no_model_amortization,
    parameter_set_amortized_bpp_table,
)


def test_amortized_rate_arithmetic_uses_stream_plus_model_share() -> None:
    assert amortized_bpp(58, 338_000, 2048, 1000) == pytest.approx(
        8.0 * (58 + 338) / 2048
    )
    table = amortized_bpp_table(58, 338_000, 2048)

    assert tuple(map(int, table)) == AMORTIZATION_CORPUS_SIZES
    assert table["128"] > table["100000"] > 8.0 * 58 / 2048


def test_model_and_no_model_accounting_share_a_machine_readable_schema() -> None:
    learned = model_amortization(58, 2048, {"fp32": 338_000, "fp16": 171_000})
    gpcc = no_model_amortization(61.25, 2048)

    assert learned["model_bytes"] == {"fp32": 338_000, "fp16": 171_000}
    assert learned["amortized_bpp"]["fp16"]["672"] == pytest.approx(
        amortized_bpp(58, 171_000, 2048, 672)
    )
    assert gpcc["model_bytes"] == 0
    assert gpcc["amortized_bpp"]["2468"] == pytest.approx(8.0 * 61.25 / 2048)


def test_gpcc_parameter_sets_are_amortized_but_side_information_is_not() -> None:
    table = parameter_set_amortized_bpp_table(
        61.25,
        29.0,
        2048,
        per_object_side_information_bytes=8,
    )

    assert table["128"] == pytest.approx(8.0 * (61.25 - 29 + 29 / 128 + 8) / 2048)
    assert table["100000"] > 8.0 * (61.25 - 29 + 8) / 2048


@pytest.mark.parametrize("corpus_size", [0, -1, True])
def test_amortized_rate_rejects_invalid_corpus_sizes(corpus_size: int) -> None:
    with pytest.raises(ValueError, match="corpus_size"):
        amortized_bpp(50, 1000, 2048, corpus_size)
