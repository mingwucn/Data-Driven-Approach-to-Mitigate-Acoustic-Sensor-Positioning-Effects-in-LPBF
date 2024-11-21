@echo off
setlocal enabledelayedexpansion

set "TOPIC=CNN_1d_cross_validation_with_differnt_folds_for_context_classification"
set "WORKDIR=$WORKDIR/MuSIC/Mitigate"
set "server=barabas"
set "batch_size=1024"
set "epochs=50"
set "folds=10"
set "repeat=1"

set "ARG1=--gpu=2 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='mic' --output_type='direction'"
set "ARG2=--gpu=2 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='mic+energy' --output_type='direction'"
set "ARG3=--gpu=2 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='mic' --output_type='position'"
set "ARG4=--gpu=2 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='mic+energy' --output_type='position'"
set "ARG5=--gpu=3 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='ae' --output_type='direction'"
set "ARG6=--gpu=3 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='ae+energy' --output_type='direction'"
set "ARG7=--gpu=3 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='ae' --output_type='position'"
set "ARG8=--gpu=3 --save_every=10 --fold_i=0 --folds=folds --model_name=CNN --batch_size=!batch_size! --learning_rate=5e-4 --repeat=repeat --epochs=epochs --num_workers=12 --input_type='ae+energy' --output_type='position'"

set "TOPICI=!TOPIC!_1_4"
set "cmd="
for /l %%i in (1,1,2) do (
    set "cmd=!cmd! torchrun train_dist_1d.py !ARG%%i! &&"
)
set "cmd=!cmd:~1,-2!"
set "GIT=git add . && git commit -am \"!TOPICI! From !server!\" && git push"
echo Starting on !server! with !TOPICI!: 
echo !cmd!
ssh -t !server! "tmux new -s !TOPICI! -n 0 -d"
@REM ssh -t !server! "tmux new -s gpu -d"
@REM ssh -t !server! "tmux new -s cpu -d"
@REM ssh !server! "tmux send -t gpu 'clear && watch nvidia-smi ' C-m"
@REM ssh !server! "tmux send -t cpu 'clear && htop ' C-m"
ssh !server! "tmux send -t !TOPICI!:0 'cd !WORKDIR! && !GIT!' C-m"
ssh !server! "tmux send -t !TOPICI!:0 'clear && cd !WORKDIR! && !cmd! && cd ./lfs && !GIT!' C-m"

set "TOPICI=!TOPIC!_5_8"
set "cmd="
for /l %%i in (3,1,4) do (
    set "cmd=!cmd! python train_dist_1d.py !ARG%%i! &&"
)
set "cmd=!cmd:~1,-2!"
set "GIT=git add . && git commit -am \"!TOPICI! From !server!\" && git push"
echo Starting on !server! with !TOPICI!: 
echo !cmd!
ssh -t !server! "tmux new -s !TOPICI! -n 0 -d"
@REM ssh -t !server! "tmux new -s gpu -d"
@REM ssh -t !server! "tmux new -s cpu -d"
@REM ssh !server! "tmux send -t gpu 'clear && watch nvidia-smi ' C-m"
@REM ssh !server! "tmux send -t cpu 'clear && htop ' C-m"
ssh !server! "tmux send -t !TOPICI!:0 'cd !WORKDIR! && !GIT!' C-m"
ssh !server! "tmux send -t !TOPICI!:0 'clear && cd !WORKDIR! && !cmd! && cd ./lfs && !GIT!' C-m"

echo Command executed on !server!


endlocal
