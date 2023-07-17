#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=working/training/log_train_%j.out
#SBATCH --job-name=training

conda activate adm

ROOT_DIR="/home/Student/s4623598/weijiang/_Miccai/PPN_new"
cd $ROOT_DIR


python -m scripts.image_train --work_dir $ROOT_DIR/working/training --dataset_name fastmri_knee --lr 1e-4 --image_size 320 --channel_mult 1,2,2,4,4,4 --attention_resolutions 20,10 --num_channels 32 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --noise_schedule cosine --learn_sigma True --diffusion_steps 1000 --class_cond False --log_interval 100 --keep_checkpoint_num 5 --save_interval 10000 --batch_size 2 



