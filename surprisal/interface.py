"""Defines the API for this module"""

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import typing
import tokenizers.Encoding


class Model(ABC):
    def __init__(self, model_id=None) -> None:
        super().__init__()
        self.model_id = model_id

    @abstractmethod
    def surprise(self, textbatch: typing.Union[typing.List, str]) -> "Surprisal":
        raise NotImplementedError


class SurprisalQuantity(float):
    def __init__(self, value, text="") -> None:
        float.__init__(value)
        self.text = text

    def __new__(self, value, text):
        return float.__new__(self, value)

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.text


class SurprisalArray(ABC):
    def __index__(self):
        pass

    def __len__(self):
        return len(self.surprisals)

    def __repr__(self) -> str:
        """
        nicely formatted surprisal string with corresponding tokens/substrings
        that are sliced into using the `__getitem__` method
        """
        numfmt = "{: >10.3f}"
        strfmt = "{: >10}"
        accumulator = ""
        for t in self.tokens:
            accumulator += strfmt.format(t[:10]) + " "
        accumulator += "\n"
        for s in self.surprisals:
            accumulator += numfmt.format(s) + " "
        return accumulator

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
        raise NotImplementedError

    @property
    @abstractmethod
    def tokens(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def surprisals(self) -> typing.Collection[SurprisalQuantity]:
        raise NotImplementedError

    def lineplot(self, f=None, a=None, cumulative=False):
        # import plotext as plt
        from matplotlib import pyplot as plt
        import numpy as np

        if f is None or a is None:
            f, a = plt.subplots()

        arr = np.cumsum(self.surprisals) if cumulative else self.surprisals
        a.plot(
            arr + np.random.rand(len(self)) / 10,
            ".--",
            lw=2,
            label=" ".join(self.tokens),
            alpha=0.9,
        )
        a.set(
            xticks=range(0, len(self.tokens)),
            xlabel=("tokens"),
            ylabel=(
                f"{'cumulative ' if cumulative else ''}surprisal (natural log scale)"
            ),
        )
        # plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left")
        plt.tight_layout()
        a.grid(visible=True)

        for i, (t, y) in enumerate(self):
            a.annotate(t, (i, arr[i]))

        return f, a


class CustomEncoding:
    """
    a duck-typed clone of the huggingface tokenizers' return class
        `tokenizers.Encoding`
    that packages simple custom-tokenized text together with its
    character and word spans allowing indexing into the tokenized
    object by character and word spans

    the goal is for this class to be capable of being passed to
    `hf_pick_matching_token_ixs` with the signature
    ```python
    surprisal.utils.hf_pick_matching_token_ixs(
        encoding: "tokenizers.Encoding", span_of_interest: slice, span_type: str
    ) -> slice
    ```
    and that's about it. it does not provide implementations of anything else,
    since huggingface makes it really difficult to actually re-use any of the
    Rust implementation of tokeizers in Python

    Arguments:
    ----------
    `tokens` (typing.Iterable[str]): the tokens in the tokenized text
    `spans` (typing.Iterable[typing.Tuple[int]]): the character spans of each token
    `original_str` (str): the original string that was tokenized

    E.g., the input to tokens and spans would be the result of the following output from
    `tokenizers.pre_tokenizers.Whitespace().pre_tokenize_str("hi my name is language model")`:
        [('hi', (0, 2)),
        ('my', (3, 5)),
        ('name', (6, 10)),
        ('is', (11, 13)),
        ('language', (14, 22)),
        ('model', (23, 29))]
    """

    def __init__(
        self,
        tokens: typing.Iterable[str],
        spans: typing.Iterable[typing.Tuple[int]],
        original_str: str,
        ids: typing.Iterable[int] = None,
    ) -> None:
        self.tokens = tokens
        self.spans = spans
        self.original_str = original_str
        self.ids = ids

    def token_to_chars(self, token_index) -> typing.Tuple[int, int]:
        """
        Get the offsets of the token at the given index.

        The returned offsets are related to the input sequence that contains the
        token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`Tuple[int, int]`: The token offsets :obj:`(first, last + 1)`
        """
        return self.spans[token_index]

    def token_to_word(self, token_index):
        """
        Get the index of the word that contains the token in one of the input sequences.

        The returned word index is related to the input sequence that contains
        the token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`int`: The index of the word in the relevant input sequence.
        """
        # assuming this is going to be primarily used for whitespace-tokenized text
        # TODO: this method will need to be fleshed out using the character spans to
        # match the tokens to their corresponding words if we ever want to support a
        # custom tokenization scheme that isn't just whitespace.
        # this is possible, but will skip implementing for now
        return token_index

    @property
    def ids(self):
        """
        The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`List[int]`: The list of IDs
        """
        # IDs are not applicable to non-LM tokenization, unless explicitly specified
        if self.ids:
            return self.ids
        return [0] * len(self.tokens)
