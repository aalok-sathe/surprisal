"""
a test script to test MWEs in the `surprisal` module
"""
from matplotlib import pyplot as plt

from surprisal import AutoHuggingFaceModel

g = AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
b = AutoHuggingFaceModel.from_pretrained(model_id="bert-base-uncased")


surpgen = g.surprise(
    [
        "The cat sat on the mat.",
        "The cat sat on the pizza.",
        "How likely is a spicy donkey?",
        "How likely is a spicy clock?",
        "How likely is a spicy dish?",
        "How likely is a spicy computer?",
        "How likely is a spicy burrito?",
    ]
)


f, a = plt.subplots()

for surp in surpgen:
    print(surp)

    surp.lineplot(
        f,
        a,
        # cumulative=True
    )
    # break

plt.show()
