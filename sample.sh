#!/bin/bash
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

conda activate adm

ROOT_DIR="/home/Student/s4623598/weijiang/_Miccai/PPN_new"
cd $ROOT_DIR


function sample_BraTS {
        echo "sample number: $1"
        echo "sample batch : $2"
        echo "sample step  : $3"
        echo "acceleration : $4"
        echo "show progress: $5"
        echo "--------------------------------------------"
        cmd="python -m scripts.image_sample  --work_dir $ROOT_DIR/working \
                --model_path $ROOT_DIR/evaluations/BraTS/model_2.02m.pt --testset_path $ROOT_DIR/evaluations/BraTS/BraTS.npz \
                --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine \
                --image_size 240 --num_channels 32 --num_res_blocks 3  --channel_mult "1,2,2,4,4" --use_ddim True \
                --num_samples $1 --batch_size $2 --timestep_respacing "ddim$3" --acceleration $4  --show_progress $5"
        echo $cmd | tr -s ' '
        echo "--------------------------------------------"
        eval $cmd
}

num=30
batch=16
step=100
for acc in 4 
do
        echo "======$acc======="
        sample_BraTS $num $batch $step $acc True
done

