"""
Tests for CausalLM-based surprisal models
"""

import pytest


@pytest.mark.parametrize("model_id", ["bert-base-uncased"])
def test_init_model(model_id):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus",
    [
        (
            "bert-base-uncased",
            "The cat sat on the mat.",
            "The mat sat on the cat.",
            0,
            float("inf"),
        )
    ],
)
def test_compute_surprisal_absolute(
    model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus
):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])

    assert abs(surp_plaus[0 : len(stim_plaus)] - expected_surp_plaus) < 1e-5
    assert abs(surp_implaus[0 : len(stim_implaus)] - expected_surp_implaus) < 1e-5


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus",
    [("bert-base-uncased", "The cat sat on the mat.", "The mat sat on the cat.")],
)
def test_compute_surprisal_relative(model_id, stim_plaus, stim_implaus):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])
    assert surp_plaus[0 : len(stim_plaus)] < surp_implaus[0 : len(stim_implaus)]
