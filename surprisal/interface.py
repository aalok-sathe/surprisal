"""Defines the API for this module"""

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import typing


class Model(ABC):
    def __init__(self, model_id=None) -> None:
        super().__init__()
        self.model_id = model_id

    @abstractmethod
    def digest(self, textbatch: typing.Union[typing.List, str]) -> "Surprisal":
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

    def lineplot(self, f=None, a=None):
        # import plotext as plt
        from matplotlib import pyplot as plt
        import numpy as np

        if f is None or a is None:
            f, a = plt.subplots()

        plt.plot(
            self.surprisals + np.random.rand(len(self)),
            ".--",
            label=" ".join(self.tokens),
        )
        plt.xticks(range(0, 1 + len(self.tokens)))  # , self.tokens)
        plt.xlabel("tokens")
        plt.ylabel("surprisal (natural log scale)")
        plt.legend()
        plt.grid()

        return f, a
