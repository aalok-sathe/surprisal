"""
Tests for MaskedLM-based surprisal models
"""

import math
import pytest


@pytest.mark.parametrize("model_id", ["bert-base-uncased"])
def test_init_model(model_id):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)


@pytest.mark.parametrize(
    "model_id, stim",
    [("bert-base-uncased", "The cat sat on the mat.")],
)
def test_compute_surprisal_sanity(model_id, stim):
    """Verify that per-token surprisals are finite, non-negative numbers."""
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp] = m.surprise([stim])

    total = surp[0 : len(stim)]
    assert math.isfinite(total), f"Expected finite total surprisal, got {total}"
    assert total > 0, f"Expected positive total surprisal, got {total}"


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus",
    [("bert-base-uncased", "The cat sat on the mat.", "The mat sat on the cat.")],
)
def test_compute_surprisal_relative(model_id, stim_plaus, stim_implaus):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])
    assert surp_plaus[0 : len(stim_plaus)] < surp_implaus[0 : len(stim_implaus)]
