#!/bin/bash
#SBATCH --job-name=figurative-nli_mistralinstruct
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/figurative-nli_mistralinstruct.log
#SBATCH --error=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/figurative-nli_mistralinstruct.err


#export CUDA_VISIBLE_DEVICES=2

export TRANSFORMERS_CACHE=".cache/huggingface/hub/"
source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

DATASET=figurative-nli
TASK=binary
PROMPT_TYPE=qa-few-ent
OUTPUT=/ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/interpretation/scripts/figurative-nli/mistralinstruct/binary_paraphrases


python3 /ikerlariak/esanchez/repositorios/metaphorLLM/zero_shot/zero_shot.py --dataset figurative-nli --model mistralinstruct --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE --paraphrases --paraphrase_source commandr
    