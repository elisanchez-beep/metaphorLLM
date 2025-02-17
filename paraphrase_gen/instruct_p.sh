#!/bin/bash
#SBATCH --job-name=figurative-nli_commandr
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/paraphrase_gen_figurative-nli_commandr.log
#SBATCH --error=.slurm/paraphrase_gen_figurative-nli_commandr.err


#export CUDA_VISIBLE_DEVICES=2
source //envs/commandr/bin/activate
export HF_HOME=.cache/huggingface/hub/


DATASET=figurative-nli
PROMPT_TYPE=instruct
OUTPUT=/metaphor_LLMs/paraphrase_gen/outputs/figurative-nli/commandr
MET_LOCATION=p
TASK=paraphrase_gen

python3 commandr.py --dataset figurative-nli --model commandr --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE --met_location $MET_LOCATION
    