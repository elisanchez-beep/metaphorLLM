#!/bin/bash

#SBATCH --job-name=qwen72_paraph
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --output=.slurm/paraph_qwen-72b.log
#SBATCH --error=.slurm/paraph_qwen-72b.err
#SBATCH --time=00:10:00
#SBATCH --mail-type=REQUEUE

source /scratch/esanchez/envs/seq-llms/bin/activate

TASK=binary
PROMPT_TYPE=qa-few-ent
OUTPUT=metaphor_LLMs/zero_shot/interpretation/outputs
MODEL=qwen-72b
PARAPH_SOURCE=commandr



export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 zero_shot_vllm.py --model $MODEL --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE --paraphrases --paraphrase_source $PARAPH_SOURCE
    