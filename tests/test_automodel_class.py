"""
Tests for automodel classes
"""

import pytest


@pytest.mark.parametrize("model_id", ["gpt2"])
def test_auto_huggingface_init(model_id):
    import surprisal

    m = surprisal.AutoHuggingFaceModel.from_pretrained(model_id=model_id)
