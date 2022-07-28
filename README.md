# surprisal
Compute surprisal from language models!

The snippet below computes per-token surprisals for a list of sentences
```python
from surprisal import AutoHuggingFaceModel

sentences = [
    "The cat is on the mat",
    "The cat is on the hat",
    "The cat is on the pizza",
    "The pizza is on the mat",
    "I told you that the cat is on the mat",
    "I told you the cat is on the mat",
]

m = AutoHuggingFaceModel.from_pretrained('gpt2')
for result in m.surprise(sentences):
    print(result)
```
and outputs the following:
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

You can also call `Surprisal.lineplot()` to visualize the surprisals:

```python
from matplotlib import pyplot as plt

f, a = None, None
for result in m.surprise(sentences):
    f, a = result.lineplot(f, a)

plt.show()
```

![](https://i.imgur.com/HusVOUq.png)


## installing
`pip install surprisal`