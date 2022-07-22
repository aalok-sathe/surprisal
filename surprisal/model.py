import typing

import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from surprisal.utils import pick_matching_token_ixs
from surprisal.interface import Model, Surprisal


class HuggingFaceSurprisal(Surprisal):
    def __init__(
        self,
        tokens: "Encoding",
        surprisals: np.ndarray,
    ) -> None:
        super().__init__()

        self._tokens: "Encoding" = tokens
        self._surprisals = surprisals

    @property
    def tokens(self):
        return self._tokens.tokens

    @property
    def surprisals(self):
        return self._surprisals

    def __iter__(self) -> typing.Tuple[str, float]:
        return zip(self.tokens, self.surprisals)

    def __getitem__(self, key):
        ixs = pick_matching_token_ixs(self._tokens, key)
        return self.surprisals[ixs]

    def __str__(self) -> str:
        numfmt = "{: >10.3f}"
        strfmt = "{: >10}"
        accumulator = ""
        for t in self.tokens:
            accumulator += strfmt.format(t[:10]) + " "
        accumulator += "\n"
        for s in self.surprisals:
            accumulator += numfmt.format(s.item()) + " "
        return accumulator


class HuggingFaceModel(Model):
    def __init__(self, model_id=None, model_class: str = None) -> None:
        super().__init__(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_class = model_class or ""
        if "gpt" in model_class + model_id.lower():
            self.model_class = AutoModelForCausalLM
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "bert" in model_class + model_id.lower():
            self.model_class = AutoModelForMaskedLM
        else:
            raise ValueError(
                f"unable to determine appropriate model class based for model_id="
                f'"{model_id}" and model_class="{model_class}". Please explicitly pass "gpt" or "bert" as model_class.'
            )

        self.model = self.model_class.from_pretrained(self.model_id)
        self.model.eval()

    def digest(self, textbatch: typing.Union[typing.List, str]) -> HuggingFaceSurprisal:
        import torch

        if type(textbatch) is str:
            textbatch = [textbatch]

        tokenized = self.tokenizer(
            textbatch,
            padding=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=True,
        )

        ids_with_bos_token = torch.concat(
            (
                torch.tensor([self.tokenizer.bos_token_id]).view(1, -1).repeat(6, 1),
                tokenized.input_ids,
            ),
            dim=1,
        )

        with torch.no_grad():
            output = self.model(
                ids_with_bos_token,
                return_dict=True,
            )
        # b, n, V
        logits = output["logits"]
        # logits[:, :, self.tokenizer.pad_token_id] = -float("inf")
        logsoftmax = torch.log_softmax(logits, dim=2)
        # here, we pick one before the current word to get surprisal of the current word
        # in context of the previous word. otherwise we would be reading off the surprisal
        # of current word given the current word plus context, which would always be high.
        logprobs = (
            logsoftmax[:, :-1, :].gather(2, tokenized.input_ids.unsqueeze(2)).squeeze(2)
        )

        for b in range(logits.shape[0]):
            yield HuggingFaceSurprisal(tokens=tokenized[b], surprisals=-logprobs[b, :])
