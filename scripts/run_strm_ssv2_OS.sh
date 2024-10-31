#!/bin/bash
#SBATCH --job-name=strm
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --time=2-00:00:00

NEW_HOME="."
COMMAND_TO_RUN="python3 run.py 
                        -c runs_strm/checkpoint_dir_ssv2_OS/ 
                        -r
                        --query_per_class 4 
                        --shot 5 
                        --way 5 
                        --trans_linear_out_dim 1152 
                        --tasks_per_batch 16 
                        --test_iters 200000 300000 400000 500000
                        --dataset ssv2 
                        --split 7 
                        -lr 0.001 
                        --method resnet50 
                        --img_size 224 
                        --scratch new 
                        --num_gpus 4 
                        --print_freq 1 
                        --save_freq 10000 
                        --training_iterations 500010 
                        --temp_set 2
                        --use_fine_grain_tasks 0.
                        --open_set"

echo ""
echo "Date = $(date)"
echo "Hostname = $(hostname -s)"
echo "Working Directory = $NEW_HOME"
echo "Command = $COMMAND_TO_RUN"
echo ""

cd $NEW_HOME
$COMMAND_TO_RUN

