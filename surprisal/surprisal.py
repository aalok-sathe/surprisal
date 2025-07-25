"""
This module implements the SurprisalArray interface, which is the main container class
for outputs produced by models.

A `SurprisalArray` is a container for a sequence of surprisal values, each associated with
a token in the input sequence. In the case of HuggingFace models, the tokens are
defined using the model's tokenizer. In the case of n-gram models, the tokens are
typically the result of whitespace-tokenization. Whitespace-tokenized text is packaged
as a `CustomEncoding` object, which tries to duck-type a HuggingFace `Encoding` object
for all relevant methods needed in `surprisal`.
"""

import typing
import logging
from functools import partial

import numpy as np
from surprisal.utils import hf_pick_matching_token_ixs
from surprisal.interface import CustomEncoding, SurprisalArray, SurprisalQuantity
import tokenizers

logger = logging.getLogger(name="surprisal")


###############################################################################
### surprisal container classes
###############################################################################


class HuggingFaceSurprisal(SurprisalArray):
    """
    Container class for surprisal values produced by HuggingFace models.
    """

    def __init__(
        self,
        tokens: "tokenizers.Encoding",
        surprisals: np.ndarray,
    ) -> None:
        super().__init__()

        self._tokens: "tokenizers.Encoding" = tokens
        self._surprisals = surprisals.astype(SurprisalQuantity)

    @property
    def tokens(self):
        return self._tokens.tokens

    @property
    def surprisals(self) -> np.typing.NDArray[SurprisalQuantity]:
        return self._surprisals

    def __iter__(self) -> typing.Tuple[str, float]:
        return zip(self.tokens, self.surprisals)

    def __getitem__(
        self, slctup: typing.Tuple[typing.Union[slice, int], str]
    ) -> SurprisalQuantity:
        """Returns the aggregated surprisal over a character

        Args:
            slctup (typing.Tuple[typing.Union[slice, int], str]):
                `(slc, slctype) = slctup`: a tuple of a `slc` (slice) and a `slctype` (str).
                `slc` gives the slice of the original string we want to aggregate surprisal over.
                `slctype` indicates if it should be a "char" slice or a "word" slice.
                if a character falls inside a token, then that entire token is included.

        Returns:
            float: the aggregated surprisal over the word span
        """
        try:
            slc, slctype = slctup
            if slctype not in ("word", "char"):
                raise ValueError(f"unrecognized slice type {slctype}")
        except TypeError:
            slc, slctype = slctup, "char"

        if slctype == "char":
            fn = partial(hf_pick_matching_token_ixs, span_type="char")
        else:  # if slctype == "word": # we already did error handling up above
            fn = partial(hf_pick_matching_token_ixs, span_type="word")

        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        token_slc = fn(self._tokens, slc)
        return SurprisalQuantity(
            self.surprisals[token_slc].sum(), " ".join(self.tokens[token_slc])
        )


class NGramSurprisal(HuggingFaceSurprisal):
    """
    Container class for surprisal values produced by n-gram models.
    """

    def __init__(
        self,
        tokens: typing.List[CustomEncoding],
        surprisals: np.ndarray,
    ) -> None:
        super().__init__(tokens, surprisals.astype(SurprisalQuantity))

    def __getitem__(
        self, slctup: typing.Tuple[typing.Union[slice, int], typing.Optional[str]]
    ):
        """Returns the aggregated surprisal over a character

        Args:
            slctup (typing.Tuple[typing.Union[slice, int], str]):
                `(slc, slctype) = slctup`: a tuple of a `slc` (slice) and a `slctype` (str).
                `slc` gives the slice of the original string we want to aggregate surprisal over.
                `slctype` indicates if it should be a "char" slice or a "word" slice.
                if a character falls inside a token, then that entire token is included.

        Returns:
            float: the aggregated surprisal over the word span
        """
        try:
            slc, slctype = slctup
            if slctype not in ("word", "char"):
                raise ValueError(f"unrecognized slice type {slctype}")
        except TypeError:
            # slctup is not a tuple, but just a slice or int
            slc, slctype = slctup, "char"

        if slctype == "char":
            raise NotImplementedError(
                'NGramSurprisal currently only supports "word" spans'
            )
            # fn = partial(hf_pick_matching_token_ixs, span_type="char")
        if slctype == "word":
            token_slc = slc
        else:
            token_slc = None

        if isinstance(slc, int):
            slc = slice(slc, slc + 1)

        return SurprisalQuantity(
            self.surprisals[token_slc].sum(), " ".join(self.tokens[token_slc])
        )


# class PCFGSurprisal(SurprisalArray):
#     ...
