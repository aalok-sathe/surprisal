"""
a test script to test MWEs in the `surprisal` module
"""
from matplotlib import pyplot as plt

import surprisal

# m = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
m = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="bert-base-uncased")


stims = [
    "The hooligan wrecked my new car!",
    "I am a cat on the mat",
    # "The cat sat on the mat.",
    # "The cat sat on the pizza.",
    # "How likely is a spicy donkey?",
    # "How likely is a spicy clock?",
    # "How likely is a spicy dish?",
    # "How likely is a spicy computer?",
    # "How likely is a spicy burrito?",
]

surps = [*m.surprise(stims), *m.surprise(stims, use_bos_token=False)]


f, a = plt.subplots()

for surp in surps:
    print(surp)

    surp.lineplot(
        f,
        a,
        # cumulative=True
    )
    # break

plt.show()

*_, surp = surps
print(f"tokens: {surp}")

for wslc in [0, 1, slice(0, 1)]:
    print(f"span of interest (word index): {wslc}")
    print(f"recovered surprisal: {surp[wslc, 'word']}")
    print("=" * 32)

pass
