
from datetime import datetime
import os
import json

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

bl = "\\"

with open("/gaueko0/users/esanchez/metaphor_LLMs/config.json", "r") as f:
    config = json.load(f)

datasets = ["meta4xnli"]
models = ["mistralinstruct", "commandr"]
met_location = "p"
task = "paraphrase_gen"

for dataset in datasets:
    for model in models:
        for prompt in config["prompts"][task]:
            scripts_path = f"/gaueko0/users/esanchez/metaphor_LLMs/{task}/scripts/{dataset}/{model}/"
            output_path = f"/gaueko0/users/esanchez/metaphor_LLMs/{task}/outputs/{dataset}/{model}"
            if not os.path.exists(scripts_path):
                os.makedirs(scripts_path)
            with open(os.path.join(scripts_path, f"{prompt}_{met_location}.sh"), "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name={dataset}_{model}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/{task}_{dataset}_{model}.log
#SBATCH --error=.slurm/{task}_{dataset}_{model}.err


#export CUDA_VISIBLE_DEVICES=2
source /gaueko0/users/esanchez/envs/commandr/bin/activate
export HF_HOME=.cache/huggingface/hub/


DATASET={dataset}
PROMPT_TYPE={prompt}
OUTPUT={output_path}
MET_LOCATION={met_location}
TASK={task}

python3 {"commandr" if model == "commandr" else 'generate_paraphrase'}.py --dataset {dataset} --model {model} --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE {"--met_location $MET_LOCATION" if dataset != "meta4xnli" else ''}
    """)          