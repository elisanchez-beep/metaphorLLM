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
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
import os

MAX_NEW_TOKENS = 5
TEMPERATURE = 0.3







def parse_args():
    os.environ['TRANSFORMERS_CACHE'] = '.cache/huggingface/hub'

    parser = argparse.ArgumentParser(
            description="Evaluate a transformers model on a text classification task"
        )

    parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="Model name in config", 
    choices=["llama3instruct","llama3-70b", "qwen-7b", "qwen-72b", "gemma4", "gemma27"]
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

    parser.add_argument(
    "--lang",
    type=str,
    default=None,
    required=False,
    help="Language to filter dataframe samples"
    )
    
    args = parser.parse_args()
    
    return args

def load_dataset(data_path: str, lang=None) -> pd.DataFrame:
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
    
    if lang:
        filtered_df = df[df["language"] == lang]
        print(len(filtered_df))
        return filtered_df
    else:
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

def format_prompts(premises, hypotheses, prompt_config, prompt_type, tokenizer):
    formatted_prompts = []
    for p, h in zip(premises, hypotheses):

        if prompt_type == "chain":
            prompt = [
            {"role": "system", "content": prompt_config.get(prompt_type, {}).get("preffix", "")},
            {"role": "user", "content": f"Premise: {p}\n Hypothesis: {h}\n Answer:"},
            ]
        else:
            prompt = [
            {"role": "system", "content": prompt_config.get(prompt_type, {}).get("preffix", "")},
            {"role": "user", "content": f" {p} -> {h}: "},
            ]

        #formatted_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        formatted_prompts.append(prompt)

    return formatted_prompts


def load_data(dataset_config, args, logger):
    
    print(args.task)
    print(dataset_config.get("prompts", []))
    
    # Ensure trilabel setup only for Meta4XNLI 
    assert args.task in dataset_config.get("prompts", []) 
    
    
    if args.paraphrases:
        data_path =  dataset_config.get("data_path_paraphrase", "")
    else:
        data_path =  dataset_config.get("data_path", "")

    logger.info(f"Dataset loaded from: {data_path}")
    df = load_dataset(data_path, args.lang)
    logger.info(f"Loaded samples: {len(df)}")
    premises = get_column_values(df, dataset_config.get("prem_col", ""))
    hypotheses = get_column_values(df, dataset_config.get("hyp_col", ""))
    logger.info(f"Loaded premises and hyp: {len(premises)}, {len(hypotheses)}")
    if args.paraphrases:
        gold_labels = get_column_values(df, "gold_label")
    else:
        gold_labels = [l.lower() for l in get_column_values(df, dataset_config.get("label_col", ""))]
    
    return premises, hypotheses, gold_labels

def process_generation_outpus(outputs_gen, logger, label_mappings):
    predictions = []

    for output in outputs_gen:
        prompt = output.prompt
        logger.info(f"Prompt: {prompt}")
        generated_text = output.outputs[0].text
        logger.info(f"Generated text: {generated_text}")
        mapped_label = map_labels(generated_text, label_mappings)
        logger.info(f"Mapped label: {mapped_label}")
        predictions.append(mapped_label)
        logger.info("Label added to predictions.")

    return predictions

def generate(model, params, dataset_name, datasets_config, prompt_config, args, tokenizer, logger):
    label_mappings = prompt_config.get(args.prompt_type, {}).get("label_mapping")


    premises, hypotheses, gold_labels = load_data(datasets_config[dataset_name], args, logger)
    formatted_prompts = format_prompts(premises, hypotheses, prompt_config, args.prompt_type, tokenizer)
    logger.info(f"Formatted prompts {formatted_prompts[:3]}")
                    
    outputs_gen = model.chat(formatted_prompts, params)

    predictions = process_generation_outpus(outputs_gen, logger, label_mappings)
    logger.info(f"Outputs: {len(predictions), type(predictions)}, outputs: {predictions[:2]}")

    logger.debug(gold_labels[:5], predictions[:5], flush=True)
    assert len(gold_labels) == len(predictions)
    logger.info(f"Gold: {len(gold_labels)}, Pred: {len(predictions)}")
    

    predictions_path = os.path.join(args.output_dir, args.model, f"{dataset_name}_{args.lang}_{args.prompt_type}{'_'+args.paraphrase_source+'_' if args.paraphrase_source else ''}{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.tsv")
    

    dump_predictions(predictions_path, premises, hypotheses, gold_labels, predictions)
    
    logger.info(f"Predictions dumped to {predictions_path}")
    
    
    
    accuracy = accuracy_score(gold_labels, predictions, normalize=True)
    logger.info(f"Accuracy {dataset_name} {len(gold_labels)}, {len(predictions)}: {accuracy}\n")

    return accuracy



def main():
    
    args = parse_args()

    with open("metaphorLLM/config.json", "r") as f:
        config = json.load(f)

    datasets_config = config.get("datasets", {})
    prompt_config = config.get("prompts", {}).get(args.task, {})

    if not os.path.exists(os.path.join(args.output_dir, args.model)):
        os.makedirs(os.path.join(args.output_dir, args.model))
        
      #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)
    
    logger_path = os.path.join(args.output_dir, args.model, f"{'_'+args.paraphrase_source+'_' if args.paraphrase_source else ''}_eval_{args.prompt_type}_{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
    
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
    logger.info(f"Language: {args.lang}")
    label_mappings = prompt_config.get(args.prompt_type, {}).get("label_mapping")
    logger.info(f"Label mappings: {label_mappings}")

    
    # LOAD MODEL

    set_seed(5)
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    logger.info("Tokenizer loaded")
    params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)
    model = LLM(model=model_id, tensor_parallel_size=4, max_model_len=8192, dtype="bfloat16", swap_space=64, gpu_memory_utilization=0.85)
    logger.info(f"Loaded model: {model_id}")
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    accuracies = {}

    for dataset_name in datasets_config:
        accuracy = generate(model, params, dataset_name, datasets_config, prompt_config, args, tokenizer, logger)    
        accuracies[dataset_name] = accuracy
        logger.info(f"Accuracy dict:  {accuracies}")
    print(f"accuracy scores: {accuracies}")
    logger.info(f"Accuracy scores: {accuracies}")
    
if __name__ == "__main__":
    main()