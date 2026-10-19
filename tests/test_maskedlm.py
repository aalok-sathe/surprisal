"""
Tests for MaskedLM-based surprisal models
"""

import math

import numpy as np
import pytest

STIM_PLAUS = "The cat sat on the mat."
STIM_IMPLAUS = "The mat sat on the cat."
# a sentence containing a word ("shimmered") that BERT's wordpiece tokenizer splits
# into several subword tokens
STIM_SUBWORD = "The telescope shimmered with glitter."


def as_floats(surp):
    """`.surprisals` is an object array of `SurprisalQuantity`; numpy needs real floats"""
    return np.asarray(surp.surprisals, dtype=float)


@pytest.mark.parametrize("model_id", ["bert-base-uncased"])
def test_init_model(model_id):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus",
    [
        (
            "bert-base-uncased",
            STIM_PLAUS,
            STIM_IMPLAUS,
            12.002234318293631,
            33.37621930241585,
        )
    ],
)
def test_compute_surprisal_absolute(
    model_id, stim_plaus, stim_implaus, expected_surp_plaus, expected_surp_implaus
):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])

    assert abs(surp_plaus[0 : len(stim_plaus)] - expected_surp_plaus) < 1e-3
    assert abs(surp_implaus[0 : len(stim_implaus)] - expected_surp_implaus) < 1e-3


@pytest.mark.parametrize(
    "model_id, stim_plaus, stim_implaus",
    [("bert-base-uncased", STIM_PLAUS, STIM_IMPLAUS)],
)
def test_compute_surprisal_relative(model_id, stim_plaus, stim_implaus):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp_plaus, surp_implaus] = m.surprise([stim_plaus, stim_implaus])
    assert surp_plaus[0 : len(stim_plaus)] < surp_implaus[0 : len(stim_implaus)]


@pytest.mark.parametrize("model_id, stim", [("bert-base-uncased", STIM_PLAUS)])
def test_special_tokens_are_nan(model_id, stim):
    """special tokens and padding carry no surprisal; everything else must be finite"""
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [surp] = m.surprise([stim])

    special = set(m.tokenizer.all_special_tokens)
    for token, value in surp:
        if token in special:
            assert math.isnan(
                value
            ), f"expected NaN for special token {token}, got {value}"
        else:
            assert (
                math.isfinite(value) and value >= 0
            ), f"bad surprisal for {token}: {value}"


@pytest.mark.parametrize(
    "model_id, stim, filler",
    [
        (
            "bert-base-uncased",
            STIM_PLAUS,
            "A considerably longer sentence, which forces the shorter one to be padded.",
        )
    ],
)
def test_padding_does_not_affect_surprisal(model_id, stim, filler):
    """a text's surprisals must not depend on what else happens to be in the batch"""
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [alone] = m.surprise([stim])
    [padded, _] = m.surprise([stim, filler])

    n = len(as_floats(alone))
    assert np.allclose(
        as_floats(alone)[:n], as_floats(padded)[:n], atol=1e-3, equal_nan=True
    )
    # the padded item is right-padded out to the length of the longer one, and every
    # one of those extra positions must be NaN rather than a spurious surprisal
    assert np.all(np.isnan(as_floats(padded)[n:]))


@pytest.mark.parametrize("model_id, stim", [("bert-base-uncased", STIM_SUBWORD)])
def test_within_word_l2r_penalizes_split_words(model_id, stim):
    """
    Under the "original" PLL metric, the first subword of a multi-token word is trivially
    predictable from the remaining subwords of that same word, which collapses its surprisal
    to ~0. "within_word_l2r" masks those later subwords too (Kauf & Ivanova, 2023), so the
    word is scored strictly more conservatively.
    """
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [orig] = m.surprise([stim], pll_metric="original")
    [l2r] = m.surprise([stim], pll_metric="within_word_l2r")

    assert l2r[0 : len(stim)] > orig[0 : len(stim)]
    # tokens that are a whole word on their own are scored identically by both metrics
    single_token_words = [
        i for i, t in enumerate(orig.tokens) if t in ("the", "with", ".")
    ]
    assert np.allclose(
        as_floats(orig)[single_token_words],
        as_floats(l2r)[single_token_words],
        atol=1e-3,
    )


@pytest.mark.parametrize("model_id, stim", [("bert-base-uncased", STIM_SUBWORD)])
def test_batch_size_invariance(model_id, stim):
    """`batch_size` only chunks the forward passes; it must not change the result"""
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [small] = m.surprise([stim], batch_size=1)
    [large] = m.surprise([stim], batch_size=1024)
    assert np.allclose(as_floats(small), as_floats(large), atol=1e-3, equal_nan=True)


@pytest.mark.parametrize("model_id, stim", [("bert-base-uncased", STIM_PLAUS)])
def test_string_input_is_treated_as_singleton_batch(model_id, stim):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    [from_str] = m.surprise(stim)
    [from_list] = m.surprise([stim])
    assert np.allclose(as_floats(from_str), as_floats(from_list), equal_nan=True)


@pytest.mark.parametrize("model_id", ["bert-base-uncased"])
def test_rejects_unknown_pll_metric(model_id):
    import surprisal

    m = surprisal.MaskedHuggingFaceModel(model_id=model_id)
    with pytest.raises(ValueError):
        m.surprise([STIM_PLAUS], pll_metric="nonexistent")
