#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --job-name=trainBTS1
#SBATCH -o working/trainBTS1/train_%j.out
#SBATCH -e working/trainBTS1/train_err_%j.out

source /clusterdata/uqwjian7/.bashrc
conda activate adm

ROOT_DIR="/clusterdata/uqwjian7/PPN_new"
cd $ROOT_DIR

echo "test --use_fp16 True"

cmd="python -m scripts.image_train --work_dir $ROOT_DIR/working/trainBTS1 --dataset_name brats --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult 1,2,2,4,4 --attention_resolutions 30 --diffusion_steps 1000 --lr 2e-4 --log_interval 100 --keep_checkpoint_num 20 --snapshot_num 9 --save_interval 10000 --batch_size 16 --use_fp16 True --resume_checkpoint $ROOT_DIR/working/trainBTS1"

echo "--------------------------------------------"
echo $cmd | tr -s ' '
echo "--------------------------------------------"
eval $cmd