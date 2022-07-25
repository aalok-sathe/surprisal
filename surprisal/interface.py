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


class Surprisal(ABC):
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
    def surprisals(self):
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
