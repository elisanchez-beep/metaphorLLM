#!/bin/bash
#SBATCH --job-name=fig-qa_llama3instruct
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/fig-qa_llama3instruct.log
#SBATCH --error=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/fig-qa_llama3instruct.err


#export CUDA_VISIBLE_DEVICES=2

export TRANSFORMERS_CACHE=".cache/huggingface/hub/"
source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

DATASET=fig-qa
TASK=binary
PROMPT_TYPE=chain
OUTPUT=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/fig-qa/llama3instruct/binary


python3 /ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/zero_shot.py --dataset fig-qa --model llama3instruct --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE   
    