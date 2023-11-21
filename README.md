# surprisal
Compute surprisal from language models!

`surprisal` supports most Causal Language Models (`GPT2`- and `GPTneo`-like models) from Huggingface or local checkpoint, 
as well as `GPT3` models from OpenAI using their API! We also support `KenLM` N-gram based language models using the
KenLM Python interface.

Masked Language Models (`BERT`-like models) are in the pipeline and will be supported at a future time (see [#9](https://github.com/aalok-sathe/surprisal/pull/9)).

# [Docs](https://aalok-sathe.github.io/surprisal/surprisal.html) [![](https://github.com/aalok-sathe/surprisal/actions/workflows/docs.yml/badge.svg)](https://aalok-sathe.github.io/surprisal/surprisal.html)


# Usage

The snippet below computes per-token surprisals for a list of sentences
```python
from surprisal import AutoHuggingFaceModel, KenLMModel

sentences = [
    "The cat is on the mat",
    "The cat is on the hat",
    "The cat is on the pizza",
    "The pizza is on the mat",
    "I told you that the cat is on the mat",
    "I told you the cat is on the mat",
]

m = AutoHuggingFaceModel.from_pretrained('gpt2')
m.to('cuda') # optionally move your model to GPU!

k = KenLMModel(model_path='./literature.arpa')

for result in m.surprise(sentences):
    print(result)
for result in k.surprise(sentences):
    print(result)
```
and produces output of this sort (`gpt2`):
```
       The       Ġcat        Ġis        Ġon       Ġthe       Ġmat  
     3.276      9.222      2.463      4.145      0.961      7.237  
       The       Ġcat        Ġis        Ġon       Ġthe       Ġhat  
     3.276      9.222      2.463      4.145      0.961      9.955  
       The       Ġcat        Ġis        Ġon       Ġthe     Ġpizza  
     3.276      9.222      2.463      4.145      0.961      8.212  
       The     Ġpizza        Ġis        Ġon       Ġthe       Ġmat  
     3.276     10.860      3.212      4.910      0.985      8.379  
         I      Ġtold       Ġyou      Ġthat       Ġthe       Ġcat        Ġis        Ġon       Ġthe       Ġmat 
     3.998      6.856      0.619      2.443      2.711      7.955      2.596      4.804      1.139      6.946 
         I      Ġtold       Ġyou       Ġthe       Ġcat        Ġis        Ġon       Ġthe       Ġmat  
     3.998      6.856      0.619      4.115      7.612      3.031      4.817      1.233      7.033 
```

## extracting surprisal over a substring

A surprisal object can be aggregated over a subset of tokens that best match a span of words or characters. 
Word boundaries are inherited from the model's standard tokenizer, and may not be consistent across models,
so using character spans when slicing is the default and recommended option.
Surprisals are in log space, and therefore added over tokens during aggregation.  For example:
```python
>>> [s] = m.surprise("The cat is on the mat")
>>> s[3:6, "word"] 
12.343366384506226
Ġon Ġthe Ġmat
>>> s[3:6, "char"]
9.222099304199219
Ġcat
>>> s[3:6]
9.222099304199219
Ġcat
```

### GPT-3 using OpenAI API

⚠ NOTE: OpenAI no longer returns log probabilities in most of their models as of recently. See [#15](https://github.com/aalok-sathe/surprisal/issues/15).
In order to use a GPT-3 model from OpenAI's API, you will need to obtain your organization ID and user-specific API key using your account.
Then, use the `OpenAIModel` in the same way as a Huggingface model.

```python
m = surprisal.OpenAIModel(model_id='text-davinci-002',
                          openai_api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
                          openai_org="org-xxxxxxxxxxxxxxxxxxxxxxxx")
```
These values can also be passed using environment variables, `OPENAI_API_KEY` and `OPENAI_ORG` before calling a script.

You can also call `Surprisal.lineplot()` to visualize the surprisals:

```python
from matplotlib import pyplot as plt
f, a = None, None
for result in m.surprise(sentences):
    f, a = result.lineplot(f, a)

plt.show()
```

![](https://i.imgur.com/HusVOUq.png)


`surprisal` also has a minimal CLI:
```python
python -m surprisal -m distilgpt2 "I went to the train station today."
      I      Ġwent        Ġto       Ġthe     Ġtrain   Ġstation     Ġtoday          . 
  4.984      5.729      0.812      1.723      7.317      0.497      4.600      2.528 

python -m surprisal -m distilgpt2 "I went to the space station today."
      I      Ġwent        Ġto       Ġthe     Ġspace   Ġstation     Ġtoday          . 
  4.984      5.729      0.812      1.723      8.425      0.707      5.182      2.574
```


# Installing
Because `surprisal` is used by people from different communities for different
purposes, by default, core dependencies related to language modeling are marked
optional. Depending on your use case, install `surprisal` with the appropriate
extras.

## Installing from PyPI (latest stable release)

Use a command like `pip install surprisal[optional]`, replacing `[optional]` with whatever optional support you need.
For multiple optional extras, use a comma-separated list:
```bash
pip install surprisal[kenlm,transformers]
```
Possible options include: `transformers`, `kenlm`, `openai`

If you use `poetry` for your existing project, use the `-E` option to add
`surprisal` together with the desired optional dependencies:
```bash
poetry add surprisal -E transformers -E openai -E kenlm
```

## Installing from GitHub (bleeding edge)

The `-e` flag allows an editable install, so you can make changes to `surprisal`.
```bash
git clone https://github.com/aalok-sathe/surprisal.git
pip install .[transformers] -e
```



# Acknowledgments

Inspired from the now-inactive [`lm-scorer`](https://github.com/simonepri/lm-scorer); thanks to
folks from [CPLlab](http://cpl.mit.edu) and [EvLab](https://evlab.mit.edu) for comments and help.


## License 
[MIT License](./LICENSE).
(C) 2022-23, contributors.
