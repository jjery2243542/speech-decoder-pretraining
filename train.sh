#!/usr/bin/bash
#SBATCH --partition=speech-gpu
##SBATCH --partition=cpu
#SBATCH -c2
##SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
##SBATCH -C 2080ti
#SBATCH -o slurm/distillation/slurm-%j.out
#SBATCH -e slurm/distillation/slurm-%j.err
#SBATCH --exclude=gpu3
#SBATCH --signal=B:10@180

source ~/.bashrc
source ~/env/activate_conda
#eval "$(conda shell.bash hook)"
conda activate tts
echo $CUDA_VISIBLE_DEVICES
data_dir=/scratch/jjery2243542/reverse_distillation/LJSpeech
data_tar_path=/share/data/speech/jjery2243542/data/LJSpeech-1.1.tar.bz2
if [[ ! -d $data_dir/LJSpeech-1.1 || ! -f $data_dir/data_finished ]]; then
    echo "untaring"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar jxf $data_tar_path -C $data_dir
    touch "$data_dir/data_finished" 
    echo "finishing untaring"
fi

data_dir=/scratch/jjery2243542/reverse_distillation/LJSpeech
data_tar_path=/share/data/speech/jjery2243542/alignment/LJSpeech/LJSpeech_tg.tar.gz
if [[ ! -d $data_dir/LJSpeech_tg || ! -f $data_dir/alignment_finished ]]; then
    echo "untaring"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar zxf $data_tar_path -C $data_dir
    touch "$data_dir/alignment_finished" 
    echo "finishing untaring"
fi

ckpt_dir=/share/data/speech/jjery2243542/reverse_distillation_ckpt/base/distillation/lr_$lr/$loss/re_init_$re_init/max_steps_$max_steps/adamW
id_dir=/share/data/speech/jjery2243542/data/LJSpeech
#conf=LJ_speech_text_hubert_base
#loss=L1
#re_init=False
#max_steps=100000
python trainer.py --data_dir $data_dir/LJSpeech-1.1/wavs --textgrid_dir $data_dir/LJSpeech_tg --conf conf/${conf}.yaml --train_id_file $id_dir/train.txt --valid_id_file $id_dir/valid.txt --save_path $ckpt_dir --n_devices 2 --override "{'optimizer': {'lr': $lr, 'loss_function': '$loss'}, 'model': {'re_init': $re_init}, 'training': {'max_steps': $max_steps}}" 
