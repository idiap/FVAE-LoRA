# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

#make sure your cache path are already setup
#otherwise HF modules would complain
model_path="vit-base-patch16-224-in21k"

fine_tuning_method="full_ft" # full_ft

lr=1e-4
batch_size=64
num_epochs=50

save_top_k=2
early_stopping_patience=5
run_test_after_train=True

check_val_every_n_epoch=1
log_every_n_steps=20
                
dataset_list=("tanganke/dtd" "tanganke/eurosat" "tanganke/gtsrb" "tanganke/resisc45" "tanganke/sun397"  "ufldl-stanford/svhn")
dataset_alias_list=("dtd" "eurosat" "gtsrb" "resisc45" "sun397" "svhn")
train_split_list=("train" "train" "train" "train" "train" "train")
val_split_list=("validation" "validation" "validation" "validation" "validation" "test")
test_split_list=("test" "test" "test" "test" "test" "test")

for seed in 1 2 42; do
for i in "${!dataset_list[@]}" ; do
    dataset_name=${dataset_list[$i]}
    dataset_alias_name=${dataset_alias_list[$i]}
    train_split_name=${train_split_list[$i]}
    val_split_name=${val_split_list[$i]}
    test_split_name=${test_split_list[$i]}

    exp_dir="exp/exp_image/${fine_tuning_method}/${model_path}/${dataset_alias_name}/${seed}"
    mkdir -p ${exp_dir}
    tensorboard_logdir="${exp_dir}/tensorboard_logs/"

    cmd="sbatch --account <your-project> --job-name "$fine_tuning_method"_"$dataset_alias_name"_"$seed"_"$model_path" --partition=gpu --gpus=1 --ntasks=1"
    cmd="${cmd} --nodes=1 --cpus-per-task=15 --time=4:00:00 --output=$exp_dir/train.%j.out --error=$exp_dir/train.%j.err"

    $cmd --wrap="python image_main.py \
        --exp-dir ${exp_dir} \
        --seed ${seed} \
        --dataset_name ${dataset_name} \
        --train_split_name ${train_split_name} \
        --val_split_name ${val_split_name} \
        --test_split_name ${test_split_name} \
        --model_path ${model_path} \
        --fine_tuning_method ${fine_tuning_method} \
        --lr ${lr} \
        --batch_size ${batch_size} \
        --num_epochs ${num_epochs} \
        --save_top_k ${save_top_k} \
        --early_stopping_patience ${early_stopping_patience} \
        --run_test_after_train  \
        --check_val_every_n_epoch ${check_val_every_n_epoch} \
        --tensorboard_use \
        --tensorboard_logdir ${tensorboard_logdir} \
        --log_every_n_steps ${log_every_n_steps}  || exit 1" 
done
done