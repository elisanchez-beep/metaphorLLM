import pandas as pd


# 1. Join met files
file_met = "repositorios/metaphorLLM/corpora/interpretation/meta4xnli/test_met.tsv"

df_met = pd.read_csv(file_met, sep="\t")

print(len(df_met))

df_all_en = df_met.loc[df_met['language'] == 'en']
print(len(df_all_en))
print(df_all_en["gold_label"])

# Check labels
assert all(df_all_en["gold_label"].isin(["entailment", "contradiction", "neutral"]))



# Dump file with original labels
assert len(df_all_en) == (len(df_met.loc[df_met['language'] == 'en']))
df_all_en.to_csv("repositorios/metaphorLLM/corpora/interpretation/meta4xnli/test_met_original_labels.tsv", sep="\t", index=False)

# Replace original labels with binary
df_all_en["gold_label"] = df_all_en["gold_label"].apply(lambda x: x.replace("contradiction", "not_entailment"))
df_all_en["gold_label"] = df_all_en["gold_label"].apply(lambda x: x.replace("neutral", "not_entailment"))
print(len(df_all_en))
print(df_all_en["gold_label"])

# Check binary labels
assert all(df_all_en["gold_label"].isin(["entailment", "not_entailment"]))
assert len(df_all_en) == (len(df_met.loc[df_met['language'] == 'en']))
df_all_en.to_csv("repositorios/metaphorLLM/corpora/interpretation/meta4xnli/test_met_binary_labels.tsv", sep="\t", index=False)


