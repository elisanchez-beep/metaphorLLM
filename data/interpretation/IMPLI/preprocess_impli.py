import pandas as pd

file_ent = "repositorios/metaphorLLM/corpora/interpretation/IMPLI/manual_e.tsv"
file_not_ent = "repositorios/metaphorLLM/corpora/interpretation/IMPLI/manual_ne.tsv"



df_ent = pd.read_csv(file_ent, sep="\t", index_col=False, header=None)
df_ent.columns = ["premise", "hypothesis"]
df_not_ent = pd.read_csv(file_not_ent, sep="\t", index_col=False, header=None)
df_not_ent.columns = ["premise", "hypothesis"]

print(len(df_ent), len(df_not_ent))

df_ent["label"] = "entailment"
df_not_ent["label"] = "not_entailment"
print(df_ent)
#df_ent.insert(-1, "label", "entailment")
#df_not_ent.insert(-1, "label", "not_entailment")

df_all = df_ent.append(df_not_ent, ignore_index = True)
print(len(df_all))
assert all(df_all["label"].isin(["entailment", "not_entailment"]))
assert len(df_all) == (len(df_ent) + len(df_not_ent))
df_all.to_csv("repositorios/metaphorLLM/corpora/interpretation/IMPLI/manual_all.tsv", sep="\t", index=False)