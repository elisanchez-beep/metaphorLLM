
from datetime import datetime
import os
import json

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

bl = "\\"

with open("/ikerlariak/esanchez/repositorios/metaphorLLM/config.json", "r") as f:
    config = json.load(f)

datasets = ["meta4xnli", "impli", "flute", "fig-qa", "figurative-nli"]
models = ["llama3instruct", "mistralinstruct"]
task = "binary"
paraphrases = True 
paraphrase_source = "mistralinstruct"

for dataset in datasets:
    for model in models:
        for prompt in config["prompts"][task]:
            if paraphrases:
                scripts_path = f"/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/{dataset}/{model}/{task}_paraphrases"
                output_path = f"/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/{dataset}/{model}/{task}_paraphrases"

            else:
                scripts_path = f"/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/{dataset}/{model}/{task}"
                output_path = f"/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/{dataset}/{model}/{task}"
            if not os.path.exists(scripts_path):
                os.makedirs(scripts_path)
            with open(os.path.join(scripts_path, f"{prompt}{'_' if paraphrase_source else ''}{paraphrase_source}.sh"), "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={dataset}_{model}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/{dataset}_{model}.log
#SBATCH --error=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/{dataset}_{model}.err


#export CUDA_VISIBLE_DEVICES=2

export TRANSFORMERS_CACHE=".cache/huggingface/hub/"
source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

DATASET={dataset}
TASK={task}
PROMPT_TYPE={prompt}
OUTPUT={output_path}


python3 /ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/zero_shot.py --dataset {dataset} --model {model} --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE {"--paraphrases" if paraphrases else ''} {"--paraphrase_source" if paraphrases else ''} {paraphrase_source if paraphrases else ''}
    """)
                            
            