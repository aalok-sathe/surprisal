import typing
import os
import logging
from abc import abstractmethod
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
)
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer

from surprisal.utils import hf_pick_matching_token_ixs, openai_models_list
from surprisal.interface import Model, SurprisalArray, SurprisalQuantity, CustomEncoding
from surprisal.surprisal import HuggingFaceSurprisal, NGramSurprisal

logger = logging.getLogger(name="surprisal")


###############################################################################
### model classes to compute surprisal
###############################################################################

# N-gram LM surprisal #
#######################


class KenLMModel(Model):
    """
    A class utilizing the `kenlm` library to compute surprisal using
    pretrained kenlm models
    """

    def __init__(self, model_path: typing.Union[str, Path], **kwargs) -> None:
        super().__init__(str(model_path))

        import kenlm  # pylint: disable=import-outside-toplevel

        self.tokenizer = Whitespace()

        self.model = kenlm.Model(model_path)
        self.state_in = kenlm.State()
        self.state_out = kenlm.State()

    def tokenize(
        self, textbatch: typing.Union[typing.List, str]
    ) -> typing.Iterator[CustomEncoding]:
        """Tokenize text using `self.tokenizer`

        Args:
            textbatch (typing.Union[typing.List, str]): str or list of strings to tokenize

        Yields:
            Iterator[CustomEncoding]: iterator over tokenized `CustomEncoding` instances
        """
        if isinstance(textbatch, str):
            textbatch = [textbatch]

        tokenized = map(self.tokenizer.pre_tokenize_str, textbatch)

        for tokens_and_spans in tokenized:
            tokens_and_spans = [*zip(*tokens_and_spans)]
            tokens = tokens_and_spans[0]
            spans = tokens_and_spans[1]
            yield CustomEncoding(tokens, spans, textbatch[0])

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        use_bos_token: bool = True,
        use_eos_token: bool = True,
    ) -> typing.List[NGramSurprisal]:
        import kenlm  # pylint: disable=import-outside-toplevel

        if isinstance(textbatch, str):
            textbatch = [textbatch]

        def score_sent(
            sent: CustomEncoding,
            m: kenlm.Model = self.model,  # pylint: disable=c-extension-no-member
        ) -> np.typing.NDArray[float]:
            st1, st2 = kenlm.State(), kenlm.State()
            if use_bos_token:
                m.BeginSentenceWrite(st1)
            else:
                m.NullContextWrite(st1)
            words = sent.tokens
            accum = []
            for w in words:
                accum += [m.BaseScore(st1, w, st2)]
                st1, st2 = st2, st1
            if use_eos_token:
                accum += [m.BaseScore(st1, "</s>", st2)]
            return np.array(accum)

        tokenized = [*self.tokenize(textbatch)]
        scores = [*map(score_sent, tokenized)]

        accumulator = []
        for b in range(len(textbatch)):
            accumulator += [NGramSurprisal(tokens=tokenized[b], surprisals=-scores[b])]
        return accumulator


# neural network LM surprisal #
###############################


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
        precision: str = "fp32",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(model_id)
        import torch  # pylint: disable=import-outside-toplevel

        precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in precisions:
            raise ValueError(
                f"precision must be one of {list(precisions.keys())}, got {precision}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # self.model_class = model_class
        self.model: PreTrainedModel = model_class.from_pretrained(
            self.model_id,
            torch_dtype=precisions[precision],
            trust_remote_code=trust_remote_code,
        )
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
        """
        Tokenizes the input text batch using the `self.tokenizer`

        Args:
            textbatch (Union[List, str]): The input text batch to be tokenized.
            max_length (int): The maximum length of the tokenized sequences.

        Returns:
            Encoding: Output from the model's default huggingface tokenizer.
        """

        if isinstance(textbatch, str):
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
    ) -> typing.List[HuggingFaceSurprisal]:
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
        if isinstance(phrases, str):
            phrases = [phrases]
        if phrases is None:
            raise ValueError("please provide a phrase to extract the surprisal of")
        textbatch = map(lambda x: str(prefix) + str(x) + str(suffix), phrases)
        slices = map(lambda x: slice(len(prefix), len(prefix + x)), phrases)
        surprisals = self.surprise([*textbatch])
        return [surp[slc, "char"] for surp, slc in zip(surprisals, slices)]


class CausalHuggingFaceModel(HuggingFaceModel):
    """
    Subclass to handle causal (autoregressive) language models from huggingface
    """

    def __init__(self, model_id=None, **kwargs) -> None:
        if "model_class" not in kwargs:
            kwargs.update(dict(model_class=AutoModelForCausalLM))
        super().__init__(model_id, **kwargs)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        use_bos_token=True,
    ) -> typing.List[HuggingFaceSurprisal]:
        """provides a measure of surprisal for `textbatch`

        Args:
            textbatch (typing.Union[typing.List, str]): either a single string or a list-like of
                strings (batch).
            use_bos_token (bool, optional): Whether the `bos_token` of tokenizer should be used and
                attached ahead of the tokenized sequence. Must be True in order to extract beginning
                token surprisal. Defaults to True.

        Returns:
            typing.List[HuggingFaceSurprisal]: a list of `HuggingFaceSurprisal` instances. each list
                item corresponds to one input in `textbatch`.
        """
        import torch  # pylint: disable=import-outside-toplevel

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
            mask = torch.concat(
                (
                    # TODO: need to evaluate what happens if this is set to 0 for the BOS token
                    torch.tensor([1])
                    .view(1, -1)
                    .repeat(tokenized.input_ids.shape[0], 1),
                    tokenized.attention_mask,
                ),
                dim=1,
            )
            # raise NotImplementedError
        else:
            ids = tokenized.input_ids
            mask = tokenized.attention_mask

        ids = ids.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )
        tokenized = tokenized.to(self.device)

        # b, n, V: batch, sequence length, vocabulary size
        logits = output["logits"]
        b, n, V = logits.shape
        # we don't want the pad token to shift the probability distribution,
        # so we set its weight to -inf
        logits[:, :, self.tokenizer.pad_token_id] = -float("inf")
        logsoftmax = torch.log_softmax(logits, dim=2)

        # for CausalLMs, we pick one before the current word to get surprisal of the current word in
        # context of the previous word. otherwise we would be reading off the surprisal of current
        # word given the current word plus context, which would always be high due to non-repetition
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
            logprobs = torch.concat(
                ((torch.ones(b, 1) * torch.nan).to(self.device), logprobs), dim=1
            )

        # b stands for an individual item in the batch; each sentence is one item
        # since this is an autoregressive model
        accumulator = []
        for b in range(logprobs.shape[0]):
            accumulator += [
                HuggingFaceSurprisal(
                    tokens=tokenized[b],
                    surprisals=-logprobs[b, :].cpu().float().numpy(),
                )
            ]
        return accumulator


class DistributedBloomModel(CausalHuggingFaceModel):
    """
    We inherit from `CausalHuggingFaceModel` since the surprisal computation is exactly the
    same, however, we pass in a different model class to support the `petals` library and use
    the BitTorrent-style distributed `bigscience/bloom-petals` model (and similar ones).
    """

    def __init__(self, model_id=None) -> None:
        """
        Construct a `DistributedBloomModel` instance
        """
        # TODO: make optional dependency group for `petals` once we flesh this out
        from petals import (
            DistributedBloomForCausalLM,
        )  # pylint: disable=import-outside-toplevel

        super().__init__(model_id, model_class=DistributedBloomForCausalLM)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: this model class is WIP
        raise NotImplementedError


class MaskedHuggingFaceModel(HuggingFaceModel):
    def __init__(self, model_id=None) -> None:
        super().__init__(model_id, model_class=AutoModelForMaskedLM)

    def surprise(
        self,
        textbatch: typing.Union[typing.List, str],
        bidirectional=False,
        fixed_length=False,
    ) -> HuggingFaceSurprisal:
        import torch  # pylint: disable=import-outside-toplevel

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

        raise NotImplementedError


# NOTE: DEPRECATED
# class OpenAIModel(HuggingFaceModel):
#    """
#    A class to support using black-box language models for surprisal
#    through the OpenAI API (GPT3 family of models). These models have
#    a different method of obtaining surprisals, since the model is not
#    locally hosted. GPT3 uses the same tokenizer as GPT2, however,
#    so we can directly feed into HuggingFaceSurprisal and benefit from
#    the same tools as the Huggingface models to extract surprisal for
#    smaller parts of the text.
#    """
#
#    def __init__(
#        self, model_id="text-davinci-002", openai_api_key=None, openai_org=None
#    ) -> None:
#        self.OPENAI_API_KEY = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
#        if self.OPENAI_API_KEY is None:
#            raise ValueError(
#                "Error: no openAI API key provided. Please pass it in "
#                "as a kwarg (`openai_api_key=...`) or specify the environment "
#                "variable OPENAI_API_KEY"
#            )
#        self.OPENAI_ORG = openai_org or os.environ.get("OPENAI_ORG", None)
#        if self.OPENAI_ORG is None:
#            raise ValueError(
#                "Error: no openAI organization ID provided. Please pass it in "
#                "as a kwarg (`openai_org=...`) or specify the environment variable OPENAI_ORG"
#            )
#
#        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#        self.tokenizer.pad_token = self.tokenizer.eos_token
#        self.request_kws = dict(
#            engine=model_id,
#            prompt=None,
#            temperature=0.5,
#            max_tokens=0,
#            top_p=1.0,
#            frequency_penalty=0.0,
#            presence_penalty=0.0,
#            logprobs=1,
#            echo=True,
#        )
#
#    def surprise(
#        self,
#        textbatch: typing.Union[typing.List, str],
#        use_bos_token=True,
#    ) -> typing.List[HuggingFaceSurprisal]:
#        import openai  # pylint: disable=C
#
#        openai.organization = self.OPENAI_ORG
#        openai.api_key = self.OPENAI_API_KEY
#
#        if type(textbatch) is str:
#            textbatch: typing.List[str] = [textbatch]
#
#        tokenized = self.tokenizer(textbatch)
#        if use_bos_token:
#            # if using BOS token, prepend each line with the BOS token
#            textbatch = list(map(lambda s: self.tokenizer.bos_token + s, textbatch))
#
#        self.request_kws["prompt"] = textbatch
#
#        response = openai.Completion.create(**self.request_kws)
#        batched = response["choices"]
#
#        # b stands for an individual item in the batch; each sentence is one item
#        # since this is an autoregressive model
#        accumulator = []
#        for b in range(len(batched)):
#            logprobs = np.array(batched[b]["logprobs"]["token_logprobs"], dtype=float)
#            tokens = batched[b]["logprobs"]["tokens"]
#
#            assert len(tokens) == len(tokenized[b]) + use_bos_token, (
#                f"Length mismatch in tokenization by GPT2 tokenizer `{tokenized[b]}` "
#                + f"and tokens returned by OpenAI GPT-3 API `{tokens}`"
#            )
#
#            accumulator += [
#                HuggingFaceSurprisal(
#                    # we have already excluded it from the tokenized object earlier
#                    tokens=tokenized[b],
#                    # if using BOS token, exclude it
#                    surprisals=-logprobs[use_bos_token:],
#                )
#            ]
#        return accumulator


class AutoModel(Model):
    """
    Factory class for initializing surprisal models based on transformers: Huggingface or OpenAI
    """

    def __init__(self) -> None:
        """
        This constructor does nothing; the correct way to use this
        class is using the `from_pretrained` classmethod as a factory to
        create instances of various model classes.
        """

    @classmethod
    def from_pretrained(
        cls, model_id_or_path, model_class: str = None, **kwargs
    ) -> typing.Union[HuggingFaceModel]:
        """
        kwargs gives the user an opportunity to specify
        the OpenAI API key and organization information
        """

        model_class = model_class or ""
        model_string = model_class.lower() + " " + model_id_or_path.lower()

        # if (
        #     "gpt3" in model_string
        #     or "openai" in model_string
        #     or model_id_or_path.lower() in openai_models_list
        # ):
        #     if "gpt3" in model_class:
        #         logger.warn(
        #             'DEPRECATION WARNING: please use "openai" as the model class. '
        #             'using "gpt3" as the model class will be deprecated in the future.'
        #         )

        #     return OpenAIModel(model_id_or_path, **kwargs)

        if "gpt" in model_string or "causal" in model_string:
            if "gpt" in model_class:
                logger.warn(
                    'DEPRECATION WARNING: please use "causal" as the model class. '
                    'using "gpt" as the model class will be deprecated in the future.'
                )
            hfm = CausalHuggingFaceModel(model_id_or_path, **kwargs)
            # for GPT-like tokenizers, pad token is not set as it is generally
            # inconsequential for autoregressive models
            hfm.tokenizer.pad_token = hfm.tokenizer.eos_token
            return hfm

        elif "bert" in model_string or "masked" in model_string:
            if "bert" in model_class:
                logger.warn(
                    'DEPRECATION WARNING: please use "masked" as the model class. '
                    'using "bert" as the model class will be deprecated in the future.'
                )
            return MaskedHuggingFaceModel(model_id_or_path)
        # in order to support the bigscience bloom-petals
        # distributed model, we make a special case.
        elif "petals" in model_string:
            hfm = DistributedBloomModel(model_id_or_path)
            # for GPT-like tokenizers, pad token is not set as it is generally
            # inconsequential for autoregressive models
            hfm.tokenizer.pad_token = hfm.tokenizer.eos_token
            return hfm
        elif (
            "kenlm" in model_string
            or model_id_or_path.endswith(".arpa")
            # this may not be a great idea since pytorch models can also end in .bin
            or model_id_or_path.endswith(".bin")
        ):
            return KenLMModel(model_id_or_path, **kwargs)
        else:
            raise ValueError(
                f"unable to determine appropriate model class based for model_id="
                f'"{model_id_or_path}" and model_class="{model_class}". '
                f'Please explicitly pass either "gpt" or "bert" as model_class.'
            )


AutoHuggingFaceModel = AutoTransformerModel = AutoModel
