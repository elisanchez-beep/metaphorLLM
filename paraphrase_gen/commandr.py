import cohere
import argparse
import json
import pathlib
from typing import List, Dict
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
import torch
import re
import tqdm
import time


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
  choices=["llama2", "llama2chat", "llama3", "llama3instruct", "mixtral8", "mistral7", "mixtralinstruct", "mistralinstruct", "commandr"]
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

def get_column_values(df, col_id):
        return df[col_id].tolist()




def dump_paraphrases(out_path: str, premises: List, hypotheses: List, gold_labels: List, paraphrases: List, met_location: str):
  assert len(paraphrases) == len(premises)
  with open(out_path, "w") as o:
    if met_location == "p":
      o.write("premise\thypothesis\tgold_label\toriginal_premise\n") 
      for p, h, g, paraph in zip(premises, hypotheses, gold_labels, paraphrases):
        o.write(f"{paraph}\t{h}\t{g}\t{p}\n")
    elif met_location == "h":
      o.write("premise\thypothesis\tgold_label\toriginal_hypothesis\n") 
      for p, h, g, paraph in zip(premises, hypotheses, gold_labels, paraphrases):
          o.write(f"{p}\t{paraph}\t{g}\t{h}\n")
  print(f"{len(paraphrases)} Predictions stored in {out_path}")

def main():
  
  co = cohere.Client(api_key="")

  with open("/ikerlariak/esanchez/repositorios/metaphorLLM/config.json", "r") as f:
    config = json.load(f)

  
  args = parse_args()
  print(args.output_dir, flush=True)
  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
      
      
  logger_path = os.path.join(args.output_dir, f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.log")
  
  logger = logging.getLogger(__name__)
  logging.basicConfig(filename=os.path.join(logger_path), encoding='utf-8', level=logging.INFO)

  model_id = config.get("models", {}).get(args.model, "")
  logger.info(f"Model used: {model_id}")
  logger.info(f"Metaphor location: {args.met_location}")
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f"Device in use: {device}")

  datasets_config = config.get("datasets", {})

  data_path =  datasets_config.get(args.dataset, {}).get("data_path", "")
  prompt_config = config.get("prompts", {}).get(args.task, {})
  preffix_prompt = prompt_config.get(args.prompt_type, {}).get("preffix", "")

  logger.info(f"Dataset loaded from: {data_path}")
  df = load_dataset(data_path)
  logger.info(f"Loaded samples: {len(df)}")
  
  
  premises = get_column_values(df, datasets_config.get(args.dataset, "").get("prem_col", ""))
  hypotheses = get_column_values(df, datasets_config.get(args.dataset, "").get("hyp_col", ""))
  gold_labels = [l.lower() for l in get_column_values(df, datasets_config.get(args.dataset, "").get("label_col", ""))]


  
 #count = 0
  
  
  
  def get_paraphrases_list(original_sentences: List[str]):
    paraphrases_sents = []
    prev_sent = None
    last_response = None
    
    for sent in original_sentences:
      logger.info(f"Sent: {sent}, prev: {prev_sent}, {sent != prev_sent}")
      if sent != prev_sent:
        prompt = preffix_prompt + f" Original sentence: {sent} Paraphrase: "
        response = co.chat(
        model=model_id,
        message=prompt)
        paraphrases_sents.append(response.text)
        logger.info(f"Prompt: {prompt}\nResponse: {response.text}")
        prev_sent = sent
        last_response = response.text
      else:
        paraphrases_sents.append(last_response)
        logger.info(f"Previous sent added: {last_response}")

    return paraphrases_sents
  
  def get_paraphrases_list_flute(original_sentences: List[str], ids_list: List[int]):
    paraphrases_sents = []
    last_id = None
    last_response = None
    for sent, idx in zip(original_sentences, ids_list):
      if idx != last_id:
        prompt = preffix_prompt + f" Original sentence: {sent} Paraphrase: "
        response = co.chat(
        model=model_id,
        message=prompt)
        paraphrases_sents.append(response.text)
        logger.info(f"Prompt: {prompt}\nResponse: {response.text}")
        last_id = idx
        last_response = response.text

      else:
        paraphrases_sents.append(last_response)
        logger.info(f"Previous sent added: {last_response}")

    return paraphrases_sents
  
  """
      count += 1
      if count % 200 == 0:
        time.sleep(65)
  """ 
      
  
  if args.met_location == "p":
    paraphrases = get_paraphrases_list(premises)
  elif args.met_location == "h":
      paraphrases = get_paraphrases_list(hypotheses)
  else:
    logger.info(f"Wrong met location: {args.met_location}")
  
  paraphrases_path = os.path.join(args.output_dir, f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}.tsv")

  dump_paraphrases(paraphrases_path, premises, hypotheses, gold_labels, paraphrases, args.met_location)
  
  

if __name__ == "__main__":
    main()