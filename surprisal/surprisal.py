import typing
import logging
from abc import abstractmethod
from functools import partial

import numpy as np
from surprisal.utils import hf_pick_matching_token_ixs
from surprisal.interface import CustomEncoding, Model, SurprisalArray, SurprisalQuantity

logger = logging.getLogger(name="surprisal")


###############################################################################
### surprisal container classes
###############################################################################


class HuggingFaceSurprisal(SurprisalArray):
    def __init__(
        self,
        tokens: "Encoding",
        surprisals: np.ndarray,
    ) -> None:
        super().__init__()

        self._tokens: "Encoding" = tokens
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
        elif slctype == "word":
            fn = partial(hf_pick_matching_token_ixs, span_type="word")

        if type(slc) is int:
            slc = slice(slc, slc + 1)

        token_slc = fn(self._tokens, slc)
        return SurprisalQuantity(
            self.surprisals[token_slc].sum(), " ".join(self.tokens[token_slc])
        )


class NGramSurprisal(HuggingFaceSurprisal):
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
            raise NotImplementedError('WIP; currently only supports "word" spans')
            fn = partial(hf_pick_matching_token_ixs, span_type="char")
        elif slctype == "word":
            token_slc = slc

        if type(slc) is int:
            slc = slice(slc, slc + 1)

        return SurprisalQuantity(
            self.surprisals[token_slc].sum(), " ".join(self.tokens[token_slc])
        )


# class PCFGSurprisal(SurprisalArray):
#     ...
