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


class CustomEncoding(tokenizers.Encoding):
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
    """

    def __init__(
        self,
        tokens: typing.Iterable[str],
        spans: typing.Iterable[typing.Tuple[int]],
        original_str: str,
    ) -> None:
        self.tokens = tokens
        self.spans = spans
        self.original_str = original_str

    def token_to_chars(self, token_index):
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

    def token_to_word(self):
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

    @property
    def ids(self):
        """
        The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`List[int]`: The list of IDs
        """
