import typing
import logging
from abc import abstractmethod
from functools import partial

import numpy as np
from surprisal.utils import pick_matching_token_ixs
from surprisal.interface import Model, SurprisalArray, SurprisalQuantity

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
    def surprisals(self):
        return self._surprisals

    def __iter__(self) -> typing.Tuple[str, float]:
        return zip(self.tokens, self.surprisals)

    def __getitem__(self, slctup: typing.Tuple[typing.Union[slice, int], str]):
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
            fn = partial(pick_matching_token_ixs, span_type="char")
        elif slctype == "word":
            fn = partial(pick_matching_token_ixs, span_type="word")

        if type(slc) is int:
            slc = slice(slc, slc + 1)

        token_slc = fn(self._tokens, slc)
        return SurprisalQuantity(
            self.surprisals[token_slc].sum(), " ".join(self.tokens[token_slc])
        )

    def __repr__(self) -> str:
        numfmt = "{: >10.3f}"
        strfmt = "{: >10}"
        accumulator = ""
        for t in self.tokens:
            accumulator += strfmt.format(t[:10]) + " "
        accumulator += "\n"
        for s in self.surprisals:
            accumulator += numfmt.format(s) + " "
        return accumulator


class PCFGSurprisal(SurprisalArray):
    ...
