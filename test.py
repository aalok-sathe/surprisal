from matplotlib import pyplot as plt
from surprisal.model import HuggingFaceModel


m = HuggingFaceModel(model_id="gpt2")


surpgen = m.digest(
    [
        "The cat sat on the mat.",
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

    surp.lineplot(f, a)
    # break

plt.show()
