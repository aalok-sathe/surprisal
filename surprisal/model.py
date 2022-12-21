import typing
import logging
from abc import abstractmethod
from functools import partial

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
)

from surprisal.utils import pick_matching_token_ixs, openai_models_list
from surprisal.interface import Model, SurprisalArray, SurprisalQuantity
from surprisal.surprisal import HuggingFaceSurprisal

logger = logging.getLogger(name="surprisal")


###############################################################################
### model classes to compute surprisal
###############################################################################
class HuggingFaceModel(Model):
    """
    A class to support language models hosted on the Huggingface Hub
    identified by a model ID
    """

    def __init__(
        self,
        model_id: str,
        model_class: typing.Callable,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # self.model_class = model_class
        self.model: PreTrainedModel = model_class.from_pretrained(self.model_id)
        self.model.eval()
        self.to(device)  # initializes a variable called `device`

    def to(self, device: str):
        """
        stateful method to move the model to specified device
        and also track device for moving any inputs
        """
        self.device = device
        self.model.to(self.device)

    def tokenize(self, textbatch: typing.Union[typing.List, str], max_length=1024):
        if type(textbatch) is str:
            textbatch = [textbatch]

        tokenized = self.tokenizer(
            textbatch,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return tokenized

    @abstractmethod
    def surprise(
        self, textbatch: typing.Union[typing.List, str]
    ) -> HuggingFaceSurprisal:
        raise NotImplementedError

    def extract_surprisal(
        self,
        phrases: typing.Union[str, typing.Collection[str]] = None,
        prefix="",
        suffix="",
    ) -> typing.List[float]:
        """
        Extracts the surprisal of the phrase given the prefix and suffix by making a call to
        `HuggingFaceSurprisal` __getitem__ object. No whitespaces or delimiters are added to
        the prefix or suffix, so make sure to provide an exact string formatted appropriately.
        """
        if type(phrases) is str:
            phrases = [phrases]
        if phrases is None:
            raise ValueError("please provide a phrase to extract the surprisal of")
        textbatch = map(lambda x: str(prefix) + str(x) + str(suffix), phrases)
        slices = map(lambda x: slice(len(prefix), len(prefix + x)), phrases)
        surprisals = self.surprise([*textbatch])
        return [surp[slc, "char"] for surp, slc in zip(surprisals, slices)]


class CausalHuggingFaceModel(HuggingFaceModel):
    def __init__(self, model_id=None) -> None:
        super().__init__(model_id, model_class=AutoModelForCausalLM)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        use_bos_token=True,
    ) -> typing.List[HuggingFaceSurprisal]:
        import torch

        tokenized = self.tokenize(textbatch)

        if use_bos_token:
            ids = torch.concat(
                (
                    torch.tensor([self.tokenizer.bos_token_id])
                    .view(1, -1)
                    .repeat(tokenized.input_ids.shape[0], 1),
                    tokenized.input_ids,
                ),
                dim=1,
            )
        else:
            ids = tokenized.input_ids

        with torch.no_grad():
            output = self.model(
                ids.to(self.device),
                return_dict=True,
            )
        tokenized = tokenized.to(self.device)

        # b, n, V
        logits = output["logits"]
        b, n, V = logits.shape
        # we don't want the pad token to shift the probability distribution,
        # so we set its weight to -inf
        logits[:, :, self.tokenizer.pad_token_id] = -float("inf")
        logsoftmax = torch.log_softmax(logits, dim=2)

        # for CausalLMs, we pick one before the current word to get surprisal of the current word in
        # context of the previous word. otherwise we would be reading off the surprisal of current
        # word given the current word plus context, which would always be high due to non-repetition.
        logprobs = (
            logsoftmax[:, :-1, :]
            .gather(
                2,
                tokenized.input_ids[:, not use_bos_token :].unsqueeze(2),
            )
            .squeeze(2)
        )
        if not use_bos_token:
            # padding to the left with a NULL because we removed the BOS token
            logprobs = torch.concat((torch.ones(b, 1) * torch.nan, logprobs), dim=1)

        # b stands for an individual item in the batch; each sentence is one item
        # since this is an autoregressive model
        accumulator = []
        for b in range(logprobs.shape[0]):
            accumulator += [
                HuggingFaceSurprisal(
                    tokens=tokenized[b], surprisals=-logprobs[b, :].cpu().numpy()
                )
            ]
        return accumulator


class MaskedHuggingFaceModel(HuggingFaceModel):
    def __init__(self, model_id=None) -> None:
        super().__init__(model_id, model_class=AutoModelForMaskedLM)

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        bidirectional=False,
        fixed_length=False,
    ) -> HuggingFaceSurprisal:
        import torch

        tokenized = self.tokenize(textbatch)
        mask_id = self.tokenizer.mask_token_id

        # BERT-like tokenizers already include a bos token in the tokenized sequence with
        # `include_special_tokens=True`
        ids_with_bos_token = tokenized.input_ids
        b, n = ids_with_bos_token.shape

        # new shape: b * n, n
        ids_with_bos_token = ids_with_bos_token.repeat(1, n - 1).view(b * (n - 1), n)
        mask_mask = torch.eye(n, n)[1:, :].repeat(b, 1).bool()
        ids_with_bos_token[mask_mask] = self.tokenizer.mask_token_id

        import IPython

        raise NotImplementedError


class OpenAIModel(HuggingFaceModel):
    """
    A class to support using black-box language models for surprisal
    through the OpenAI API (GPT3 family of models). These models have
    a different method of obtaining surprisals, since the model is not
    locally hosted. GPT3 uses the same tokenizer as GPT2, however,
    so we can directly feed into HuggingFaceSurprisal and benefit from
    the same tools as the Huggingface models to extract surprisal for
    smaller parts of the text.
    """

    def __init__(
        self, model_id="text-davinci-002", openai_api_key=None, openai_org=None
    ) -> None:
        import os

        self.OPENAI_API_KEY = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
        if self.OPENAI_API_KEY is None:
            raise ValueError(
                "Error: no openAI API key provided. Please pass it in "
                "as a kwarg (`openai_api_key=...`) or specify the environment variable OPENAI_API_KEY"
            )
        self.OPENAI_ORG = openai_org or os.environ.get("OPENAI_ORG", None)
        if self.OPENAI_ORG is None:
            raise ValueError(
                "Error: no openAI organization ID provided. Please pass it in "
                "as a kwarg (`openai_org=...`) or specify the environment variable OPENAI_ORG"
            )

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.request_kws = dict(
            engine=model_id,
            prompt=None,
            temperature=0.5,
            max_tokens=0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            echo=True,
        )

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        use_bos_token=True,
    ) -> typing.List[HuggingFaceSurprisal]:
        import openai

        openai.organization = self.OPENAI_ORG
        openai.api_key = self.OPENAI_API_KEY

        if type(textbatch) is str:
            textbatch: typing.List[str] = [textbatch]

        tokenized = self.tokenizer(textbatch)
        if use_bos_token:
            # if using BOS token, prepend each line with the BOS token
            textbatch = list(map(lambda s: self.tokenizer.bos_token + s, textbatch))

        self.request_kws["prompt"] = textbatch

        response = openai.Completion.create(**self.request_kws)
        batched = response["choices"]

        # b stands for an individual item in the batch; each sentence is one item
        # since this is an autoregressive model
        accumulator = []
        for b in range(len(batched)):
            logprobs = np.array(batched[b]["logprobs"]["token_logprobs"], dtype=float)
            tokens = batched[b]["logprobs"]["tokens"]

            assert (
                len(tokens) == len(tokenized[b]) + use_bos_token
            ), f"Length mismatch in tokenization by GPT2 tokenizer `{tokenized[b]}` and tokens returned by OpenAI GPT-3 API `{tokens}`"

            accumulator += [
                HuggingFaceSurprisal(
                    # we have already excluded it from the tokenized object earlier
                    tokens=tokenized[b],
                    # if using BOS token, exclude it
                    surprisals=-logprobs[use_bos_token:],
                )
            ]
        return accumulator


class AutoTransformerModel(Model):
    """
    Factory class for initializing surprisal models based on transformers, either Huggingface or OpenAI
    """

    def __init__(self) -> None:
        """
        this `__init__` method does nothing; the correct way to use this
        class is using the `from_pretrained` classmethod.
        """

    @classmethod
    def from_pretrained(
        cls, model_id, model_class: str = None, **kwargs
    ) -> typing.Union[HuggingFaceModel, OpenAIModel]:
        """
        kwargs gives the user an opportunity to specify
        the OpenAI API key and organization information
        """

        model_class = model_class or ""
        if (
            "gpt3" in model_class.lower() + " " + model_id.lower()
            or model_id.lower() in openai_models_list
        ):
            return OpenAIModel(model_id, **kwargs)
        elif "gpt" in model_class.lower() + " " + model_id.lower():
            hfm = CausalHuggingFaceModel(model_id)
            # for GPT-like tokenizers, pad token is not set as it is generally inconsequential for autoregressive models
            hfm.tokenizer.pad_token = hfm.tokenizer.eos_token
            return hfm
        elif "bert" in model_class.lower() + " " + model_id.lower():
            return MaskedHuggingFaceModel(model_id)
        else:
            raise ValueError(
                f"unable to determine appropriate model class based for model_id="
                f'"{model_id}" and model_class="{model_class}". '
                f'Please explicitly pass either "gpt" or "bert" as model_class.'
            )


AutoHuggingFaceModel = AutoTransformerModel
