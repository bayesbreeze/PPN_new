#!/bin/bash
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=working/sample_%j.out
#SBATCH --job-name=sample

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
                --image_size 240 --num_channels 32 --num_res_blocks 3  --channel_mult 1,2,2,4,4 --use_ddim True \
                --num_samples $1 --batch_size $2 --timestep_respacing ddim$3 --acceleration $4  --show_progress $5"
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





# ROOT_DIR="/home/Student/s4623598/weijiang/guided-diffusion"
# cd $ROOT_DIR

# python -m scripts.image_train --work_dir $ROOT_DIR/working --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult "1,2,2,4,4" --attention_resolutions "30" --diffusion_steps 1000 --lr 2e-4 --batch_size 32 --log_interval 100 --save_interval 10000 --resume_checkpoint $ROOT_DIR/working/model2020000.pt

# python -m scripts.image_train --work_dir $ROOT_DIR/working/training --dataset_name fastmri_knee --lr 1e-4 --image_size 320 --channel_mult 1,2,2,4,4,4 --attention_resolutions 20,10 --num_channels 32 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --noise_schedule cosine --learn_sigma True --diffusion_steps 1000 --class_cond True --log_interval 100 --save_interval 10000 --batch_size 128


