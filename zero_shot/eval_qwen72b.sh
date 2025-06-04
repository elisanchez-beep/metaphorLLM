#!/bin/bash
#SBATCH --job-name=qwen72
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=512GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --output=.slurm/qwen-72b.log
#SBATCH --error=.slurm/qwen-72b.err
#SBATCH --mail-type=REQUEUE


source /scratch/esanchez/envs/seq-llms/bin/activate

TASK=binary
PROMPT_TYPE=chain
OUTPUT=metaphor_LLMs/zero_shot/interpretation/outputs
MODEL=qwen-72b


export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 zero_shot_vllm.py --model $MODEL --output_dir $OUTPUT --task $TASK --prompt_type $PROMPT_TYPE 
    