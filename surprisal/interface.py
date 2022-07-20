"""Defines the API for this module"""

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import typing






class Model(ABC):

    def __init__(self, model_id=None) -> None:
        super().__init__()
        self.model_id = model_id

    def surprisal(text: str) -> 'Surprisal':
        pass



class Surprisal(ABC):

    tokens: str = None
    surprisals: typing.Any = None
    
    def __index__(self):
        pass