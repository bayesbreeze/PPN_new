#!/bin/zsh

# sh sample.sh `pwd`
ROOT_DIR=$1

function do_sample {
        echo  "step: $1"
        echo "num: $2"
        echo "acc: $3"
        echo "extra: $4"
        echo "sample type: $5"

        python -m scripts.image_sample  --work_dir $ROOT_DIR/working --model_path $ROOT_DIR/evaluations/BraTS/model_2.02m.pt \
                --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine \
                --image_size 240 --num_channels 32 --num_res_blocks 3  --channel_mult "1,2,2,4,4" \
                --batch_size $2 --num_samples $2  --timestep_respacing "ddim$1" --use_ddim True --show_progress True
}

step=10
for acc in 4 
do
        echo "======$acc======="
        do_sample $step 2 $acc 0 pp
done

