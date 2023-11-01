"""Defines the API for this module"""

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import typing


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
    hf_pick_matching_token_ixs(
        encoding: "tokenizers.Encoding", span_of_interest: slice, span_type: str
    ) -> slice
    ```
    """
