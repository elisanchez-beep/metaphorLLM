import pandas as pd

test_file = "repositorios/metaphorLLM/corpora/interpretation/FLUTE/metaphor_test.jsonl"


# Generate reformatted file with original labels

df = pd.read_json(test_file, lines=True)
df["label"] = df["label"].apply(lambda x: x.lower())
print(df)
print(len(df))
print(df["label"])
assert all(df["label"].isin(["entailment", "contradiction"]))

all_labels_path = "repositorios/metaphorLLM/corpora/interpretation/FLUTE/metaphor_test_original_labels.tsv"
df.to_csv(all_labels_path, sep="\t", index=False)
print(f"File dumped to {all_labels_path}")

# Generate reformatted file with binary labels (entailment/not_entailment)

df["label"] = df["label"].apply(lambda x: x.replace("contradiction", "not_entailment"))
print(df)
print(len(df))
print(df["label"])
assert all(df["label"].isin(["entailment", "not_entailment"]))
binary_path = "repositorios/metaphorLLM/corpora/interpretation/FLUTE/metaphor_test_binary_labels.tsv"
df.to_csv(binary_path, sep="\t", index=False)
print(f"File dumped to {binary_path}")

