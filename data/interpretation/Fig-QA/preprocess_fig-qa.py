import pandas as pd

test_file = "metaphorLLM/corpora/interpretation/Fig-QA/dev.csv"
output_path = "metaphorLLM/corpora/interpretation/Fig-QA/dev_binary_labels.tsv"
df = pd.read_csv(test_file)
print(df)

premises = df["startphrase"].to_list()
hyp1 = df["ending1"].to_list()
hyp2 = df["ending2"].to_list()
labels = df["labels"].to_list()

hypotheses = list(zip(hyp1, hyp2))


def get_sentences_by_label(hypotheses_list, label):
    hypotheses_list  = list(hypotheses_list)
    entailed = hypotheses_list[label]
    not_entailed = None 
    hypotheses_list.pop(label)
    not_entailed = hypotheses_list[0]
    
    return entailed, not_entailed
    
#print(hypotheses)
print(len(df))
preprocessed_data = []

# For each premise, extract entailed and not_entailed hyps. Generate csv with one line per prem-hyp pair.

for p, h, l in zip(premises, hypotheses, labels):
    entailed, not_entailed = get_sentences_by_label(h, l)
    #print(f"{p}, entailed: {entailed}, not_entailed:{not_entailed}, {l} ")
    preprocessed_data.append([p, entailed, "entailment"])
    preprocessed_data.append([p, not_entailed, "not_entailment"])

print(len(preprocessed_data))
assert len(preprocessed_data) == len(df)*2
df_preproc = pd.DataFrame(preprocessed_data, columns=["premise", "hypothesis", "label"])
print(df_preproc)
df_preproc.to_csv(output_path, sep="\t", index=False)
print(f"File dumped to {output_path}")
