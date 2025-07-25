"""
Tests for CausalLM-based surprisal models
"""

import pytest


@pytest.mark.parametrize("model_id", ["gpt2"])
def test_init_model(model_id):
    import surprisal

    m = surprisal.CausalHuggingFaceModel(model_id=model_id)


@pytest.mark.parametrize("model_id, stim", [("gpt2", "The cat sat on the mat.")])
def test_compute_surprisal_unconditional(model_id, stim):
    import surprisal

    m = surprisal.CausalHuggingFaceModel(model_id=model_id)
    surp = m.surprise(stim)


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus",
    [
        (
            "gpt2",
            "The cat sat on the mat.",
            "The mat sat on the cat.",
            30.357331335544586,
            40.83772802352905,
        )
    ],
)
def test_compute_surprisal_absolute(
    model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus
):
    import surprisal

    m = surprisal.CausalHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])

    assert abs(surp_plaus[0 : len(stim_plaus)] - expected_surp_plaus) < 1e-5
    assert abs(surp_implaus[0 : len(stim_implaus)] - expected_surp_implaus) < 1e-5


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus",
    [("gpt2", "The cat sat on the mat.", "The mat sat on the cat.")],
)
def test_compute_surprisal_relative(model_id, stim_plaus, stim_implaus):
    import surprisal

    m = surprisal.CausalHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])
    assert surp_plaus[0 : len(stim_plaus)] < surp_implaus[0 : len(stim_implaus)]


if __name__ == "__main__":
    test_compute_surprisal_unconditional(
        "sshleifer/tiny-gpt2", ["The cat sat.", "I am going on a bear hunt."]
    )
