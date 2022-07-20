
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from surprisal.interface import Model, Surprisal
import typing


class HuggingFaceSurprisal(Surprisal):

    def __getitem__(self, key):
        pass


class HuggingFaceModel(Model):

    def __init__(self, model_id=None, model_class: str = None) -> None:
        super().__init__(model_id)

        if model_class: pass
        if 'gpt' in model_id.lower():
            self.model_class = AutoModelForCausalLM
        elif 'bert' in model_id.lower():
            self.model_class = AutoModelForMaskedLM
        else:
            raise ValueError

        self.model = ...

    def __getitem__(self, key):
        pass