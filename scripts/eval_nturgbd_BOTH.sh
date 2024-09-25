#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --time=2-00:00:00

# NEW_HOME="./strm_ssv2/"
#  "results/testing_nturgbd_train_R_test_R/" "results/testing_nturgbd_train_R_test_FG/" 
#checkpoint_dir_nturgbd" "checkpoint_dir_nturgbd"
# False True
log_dirs=("results/testing_nturgbd_train_FG_test_FG/")
checkpoint_dirs=("runs_strm/checkpoint_dir_nturgbd_BOTH")
use_fine_grain_tasks=(0)  # probabilities of using fine-grain tasks

for i in {0..3}; do
    log_file_name=$(basename ${log_dirs[$i]})
    
    {
        echo ""
        echo "Date = $(date)"
        echo "Hostname = $(hostname -s)"
        echo "Using log_dir: ${log_dirs[$i]}"
        echo "Using checkpoint_dir: ${checkpoint_dirs[$i]}"
        echo "Using use_fine_grain_tasks: ${use_fine_grain_tasks[$i]}"
        
        COMMAND_TO_RUN="python3 run.py -c ${log_dirs[$i]} \
                        --query_per_class 4 \
                        --shot 5 \
                        --way 5 \
                        --trans_linear_out_dim 1152 \
                        --tasks_per_batch 16 \
                        --test_iters 75000 \
                        --dataset nturgbd \
                        --split 7 \
                        -lr 0.001 \
                        --method resnet50 \
                        --img_size 224 \
                        --scratch /home/sberti_datasets/ \
                        --num_gpus 4 \
                        --print_freq 1 \
                        --save_freq 75000 \
                        --training_iterations 75010 \
                        --temp_set 2 \
                        --test_model_only True \
                        --test_model_path ${checkpoint_dirs[$i]}/checkpoint60000.pt \
                        --num_test_tasks 1000"
        
        if [ "${use_fine_grain_tasks[$i]}" > 0 ]; then
            COMMAND_TO_RUN="$COMMAND_TO_RUN --use_fine_grain_tasks ${use_fine_grain_tasks[$i]}"
        fi
        
        echo "Running command: $COMMAND_TO_RUN"
        
        # Run the command
        $COMMAND_TO_RUN
    } &> "results/${log_file_name}_output_log.txt"
done