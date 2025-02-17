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
import re

MAX_NEW_TOKENS = 20
TEMPERATURE = 0.3






with open("/ikerlariak/esanchez/repositorios/metaphorLLM/config.json", "r") as f:
    config = json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(
            description="Generate literal paraphrases for nli datasets"
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
    help="Output path to dump paraphrased dataset"
    )
    
    parser.add_argument(
    "--task",
    type=str,
    default=None,
    required=True,
    help="Type of task formulation",
    choices=["paraphrase_gen"]
    )
    
    parser.add_argument(
    "--prompt_type",
    type=str,
    default=None,
    required=True,
    help="Type of prompt"
    )
    
    parser.add_argument(
    "--met_location",
    type=str,
    default=None,
    required=True,
    help="Sentence with metaphorical expressions: premise or hypothesis",
    choices=["p", "h"]
    )
    
    args = parser.parse_args()
    
    return args

def load_dataset(data_path: str) -> pd.DataFrame:
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

def dump_paraphrases(paraphrases_path: str, premises: List[str], hypotheses: List[str], paraphrases: List[str], met_location: str, gold_labels: List[str]):
    with open(paraphrases_path, "w") as o:
        if met_location == "p":
            o.write("premise\thypothesis\tgold_label\toriginal_premise\n") 
            for prem, h, g, paraph in zip(premises, hypotheses, gold_labels, paraphrases):
                o.write(f"{paraph}\t{h}\t{g}\t{prem}\n")
        elif met_location == "h":
            o.write("premise\thypothesis\tgold_label\toriginal_hypothesis\n") 
            for prem, h, g, paraph in zip(premises, hypotheses, gold_labels, paraphrases):
                o.write(f"{prem}\t{paraph}\t{g}\t{h}\n")
            
    print(f"{len(paraphrases)} Paraphrases stored in {paraphrases_path}")
    

def clean_answer(answer: str):
    answer_alpha = re.sub(r"[0-9]+.", "", answer)
    answer_dot = answer_alpha.split(".")[0]
    answer_alpha = answer_dot.strip()
    
    return answer_alpha
    


def main():
    
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logger_path = os.path.join(args.output_dir, f"{args.prompt_type}_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(logger_path), encoding='utf-8', level=logging.INFO)

    


    login(token='')
    model_id = config.get("models", {}).get(args.model, "")
    logger.info(f"Model used: {model_id}")
    logger.info(f"Prompt task: {args.task}")
    logger.info(f"Prompt config: {args.prompt_type}")
    logger.info(f"Metaphor location: {args.met_location}")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device in use: {device}")


    def get_column_values(df, col_id):
        return df[col_id].tolist()

    datasets_config = config.get("datasets", {})
    prompt_config = config.get("prompts", {}).get(args.task, {})
    
    print(args.task)
    print(datasets_config.get(args.dataset, {}).get("prompts", []))
    
    data_path =  datasets_config.get(args.dataset, {}).get("data_path", "")

    logger.info(f"Dataset loaded from: {data_path}")
    df = load_dataset(data_path)
    logger.info(f"Loaded samples: {len(df)}")
    premises = get_column_values(df, datasets_config.get(args.dataset, "").get("prem_col", ""))
    hypotheses = get_column_values(df, datasets_config.get(args.dataset, "").get("hyp_col", ""))
    gold_labels = [l.lower() for l in get_column_values(df, datasets_config.get(args.dataset, "").get("label_col", ""))]
    labels = list(set(gold_labels))

    set_seed(5)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
   
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    paraphrases = []
    
    for p, h, l in zip(premises, hypotheses, gold_labels):
        preffix_prompt = prompt_config.get(args.prompt_type, {}).get("preffix", "")
        if args.met_location == "p":
            prompt = preffix_prompt + f" Original sentence: {p} Paraphrase: "
            logger.info(f"Prompt premise sentence: {prompt}")
        elif args.met_location == "h":
            prompt = preffix_prompt + f" Original sentence: {h} Paraphrase: "
            logger.info(f"Prompt hypothesis sentence: {prompt}")
        
        
        
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        ### FOR GREEDY SEARCH
        
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, return_dict_in_generate=True, output_scores=True, temperature=TEMPERATURE)
    
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        #logger.info(f"{outputs.sequences}\t{outputs.scores}")
        
        
        #print(f"transition scores: {transition_scores}", flush=True)

        #print(f"transition scores: {transition_scores}", flush=True)
        # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        # encoder-decoder models, like BART or T5.
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        
        answers = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        logger.info(f"Answers: {answers}, split: {answers.splitlines()}")
        clean_paraphrase = clean_answer(answers.splitlines()[0])
        logger.info(f"Clean paraphrasis: {clean_paraphrase}")
        paraphrases.append(clean_paraphrase)
        logger.info("Sentence added to paraphrases.")
        

            
    logger.info(f"Paraphrases generated: {len(paraphrases)}")
    logger.info(f"Prem: {len(premises)}, hyp: {len(hypotheses)}, labels: {len(gold_labels)}, paraphrases: {len(paraphrases)}")
    paraphrases_path = os.path.join(args.output_dir, f"{args.prompt_type}_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.tsv")
    
    dump_paraphrases(paraphrases_path, premises, hypotheses, paraphrases, args.met_location, gold_labels)


    logger.info(f"Paraphrases dumped to {paraphrases_path}")
    
    


if __name__ == "__main__":
    main()