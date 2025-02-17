from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from huggingface_hub import login
import re
import sys
from sklearn.metrics import accuracy_score
import argparse
import json
import pathlib
from typing import List, Dict
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os

MAX_NEW_TOKENS = 5
TEMPERATURE = 0.3





with open("/gaueko0/users/esanchez/metaphor_LLMs/config.json", "r") as f:
    config = json.load(f)

def parse_args():
    os.environ['TRANSFORMERS_CACHE'] = '/gaueko0/users/esanchez/metaphor_LLMs/paraphrase_gen/.cache/huggingface/hub'

    parser = argparse.ArgumentParser(
            description="Finetune a transformers model on a text classification task"
        )

    parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    required=True,
    help="Name of the dataset to predict gold_labels",
    choices=["fig-qa", "figurative-nli", "impli", "flute", "meta4xnli", "meta4xnli_no_met", "meta4xnli_all"]
    )
    
    parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="Model name in config", 
    choices=["llama2", "llama2chat", "llama3", "llama3instruct", "mixtral8", "mistral7", "mixtralinstruct", "mistralinstruct"]
    )
    
    parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    required=True,
    help="Output path to dump predictions"
    )
    
    parser.add_argument(
    "--task",
    type=str,
    default=None,
    required=True,
    help="Type of task formulation",
    choices=["binary", "trilabel"]
    )
    
    parser.add_argument(
    "--prompt_type",
    type=str,
    default=None,
    required=True,
    help="Type of prompt"
    )
    
    parser.add_argument(
    "--paraphrases",
    action="store_true",
    required=False,
    help="Dataset with paraphrases generated automatically"
    )
    
    parser.add_argument(
    "--paraphrase_source",
    type=str,
    default=None,
    required=False,
    help="Model used to generate paraphrases"
    )
    
    args = parser.parse_args()
    
    return args

def load_dataset(data_path: str) -> pd.DataFrame:
    df = None
    extension = pathlib.Path(data_path).suffix
    if extension.endswith("json"):
        df = pd.read_json(data_path)
    elif extension.endswith("jsonl"):
        df = pd.read_json(data_path, lines=True)
    elif extension.endswith("tsv"):
        df = pd.read_csv(data_path, sep="\t")
    else:
        df = pd.read_csv(data_path)
        
    return df

def dump_predictions(out_path: str, premises: List, hypotheses: List, gold_labels: List, predictions: List, paraphrased_sents=None):
    if paraphrased_sents:
        with open(out_path, "w") as o:
            o.write("premise\thypothesis\tgold_label\tprediction\tparaphrased_sentence\n") 
            for p, h, g, pr, paraph in zip(premises, hypotheses, gold_labels, predictions, paraphrased_sents):
                o.write(f"{p}\t{h}\t{g}\t{pr}\t{paraph}\n")
    else:
        with open(out_path, "w") as o:
            o.write("premise\thypothesis\tgold_label\tprediction\n") 
            for p, h, g, pr in zip(premises, hypotheses, gold_labels, predictions):
                o.write(f"{p}\t{h}\t{g}\t{pr}\n")
            
    print(f"{len(predictions)} Predictions stored in {out_path}")
    
def map_labels(predictions: List[str], label_mapping: Dict):
    predictions_clean = [pred.strip(".,") for pred in predictions.lower().split()]
    for pred in predictions_clean:
        if pred in label_mapping:
            return label_mapping[pred]
    
    return "unk"
        
def get_column_values(df, col_id):
        return df[col_id].tolist()



def main():
    
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    
    logger_path = os.path.join(args.output_dir, f"{args.prompt_type}_{args.paraphrase_source+'_' if args.paraphrase_source else ''}{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(logger_path), encoding='utf-8', level=logging.INFO)

    


    login(token='')
    model_id = config.get("models", {}).get(args.model, "")
    logger.info(f"Model used: {model_id}")
    logger.info(f"Prompt task: {args.task}")
    logger.info(f"Dataset with paraphrases: {args.paraphrases}")
    logger.info(f"Prompt config: {args.prompt_type}")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device in use: {device}")


    

    datasets_config = config.get("datasets", {})
    prompt_config = config.get("prompts", {}).get(args.task, {})
    
    print(args.task)
    print(datasets_config.get(args.dataset, {}).get("prompts", []))
    
    # Ensure trilabel setup only for Meta4XNLI 
    assert args.task in datasets_config.get(args.dataset, {}).get("prompts", [])
    
    
    if args.paraphrases:
        data_path =  datasets_config.get(args.dataset, {}).get("data_path_paraphrase", "")
    else:
        data_path =  datasets_config.get(args.dataset, {}).get("data_path", "")

    logger.info(f"Dataset loaded from: {data_path}")
    df = load_dataset(data_path)
    logger.info(f"Loaded samples: {len(df)}")
    premises = get_column_values(df, datasets_config.get(args.dataset, "").get("prem_col", ""))
    hypotheses = get_column_values(df, datasets_config.get(args.dataset, "").get("hyp_col", ""))
    if args.paraphrases:
        gold_labels = get_column_values(df, "gold_label")
    else:
        gold_labels = [l.lower() for l in get_column_values(df, datasets_config.get(args.dataset, "").get("label_col", ""))]
    
    labels = list(set(gold_labels))

    set_seed(5)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions = []
    for p, h, l in zip(premises, hypotheses, gold_labels):
        preffix_prompt = prompt_config.get(args.prompt_type, {}).get("preffix", "")
        if args.prompt_type == "chain":
            prompt = preffix_prompt + f"\n Premise: {p}\n Hypothesis: {h}\n Answer: "
        else:
            prompt = preffix_prompt + f" {p} -> {h}: "
        logger.info(f"Prompt: {prompt}")
        
        
        label_mappings = prompt_config.get(args.prompt_type, {}).get("label_mapping")
        
        logger.info(f"Label mappings: {label_mappings}")
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        logger.info(f"{p}\t{h}\t{l}")
                
    
        
        
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, return_dict_in_generate=True, output_scores=True, temperature=TEMPERATURE)
    
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        logger.info(f"{outputs.sequences}\t{outputs.scores}")
        
        
        #print(f"transition scores: {transition_scores}", flush=True)

        #print(f"transition scores: {transition_scores}", flush=True)
        # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        # encoder-decoder models, like BART or T5.
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            # | token | token string | log probability | probability
            logger.info(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score}")
            #logger.info(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
            #o.write(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")


        answers = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        logger.info(f"Answers: {answers}, split: {answers.split()}")
        logger.info(f"Mapped label: {map_labels(answers, label_mappings)}")
        predictions.append(map_labels(answers, label_mappings))
        logger.info("Label added to predictions.")
        
            
    logger.debug(gold_labels[:5], predictions[:5], flush=True)
    assert len(gold_labels) == len(predictions)
    logger.info(f"Gold: {len(gold_labels)}, Pred: {len(predictions)}")
    
   
    predictions_path = os.path.join(args.output_dir, f"{args.prompt_type}_{args.paraphrase_source+'_' if args.paraphrase_source else ''}{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.tsv")
    
    if args.paraphrases:
        paraphrased_sents = df.iloc[:, -1].tolist()
        logger.info(f"Dumping predictions with paraphrased sentences, met location: {list(df.columns)[-1]}")
        dump_predictions(predictions_path, premises, hypotheses, gold_labels, predictions, paraphrased_sents)    
    else:
        dump_predictions(predictions_path, premises, hypotheses, gold_labels, predictions)
    
    logger.info(f"Predictions dumped to {predictions_path}")
    
    
    
    accuracy = accuracy_score(gold_labels, predictions, normalize=True)
    logger.info(f"Accuracy {len(gold_labels)}, {len(predictions)}: {accuracy}\n")
    


if __name__ == "__main__":
    main()