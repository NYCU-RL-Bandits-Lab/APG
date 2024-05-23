#!/usr/bin/env bash
source /home/m-ru/anaconda3/bin/activate
source /home/m-ru/anaconda3/bin/activate rl_baselines3_zoo
declare -a env_list=(
    "CarnivalNoFrameskip-v4" 
    "RiverraidNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "SeaquestNoFrameskip-v4"
    # "AsteroidsNoFrameskip-v4"
    # "BeamRiderNoFrameskip-v4"
    # "QbertNoFrameskip-v4"
    # "RoadRunnerNoFrameskip-v4"
    # "BreakoutNoFrameskip-v4"
    # "MsPacmanNoFrameskip-v4"
    # "SpaceInvadersNoFrameskip-v4"
    # "BoxingNoFrameskip-v4"
)
# echo ${env_list[*]}
for env in ${env_list[*]}
do
    # # pg
    # for seed in {0..1}
    # do
    #     ./run-pg-atari.sh $env $seed &
    #     sleep 5
    # done
    # wait
    # for seed in {2..3}
    # do
    #     ./run-pg-atari.sh $env $seed &
    #     sleep 5
    # done
    # wait
    # ./run-pg-atari.sh $env 4
    # for seed in {0..4}
    # do
    #     # apg
    #     ./run-apg-atari.sh $env $seed 0.4 &
    #     sleep 5
    #     ./run-apg-atari.sh $env $seed 0.5
    #     wait
    #     ./run-apg-atari.sh $env $seed 0.6 &
    #     sleep 5
    #     ./run-apg-atari.sh $env $seed 0.7 
    #     wait
    #     # hb
    #     ./run-hb-atari.sh $env $seed 0.4 &
    #     sleep 5
    #     ./run-hb-atari.sh $env $seed 0.5 
    #     wait
    #     ./run-hb-atari.sh $env $seed 0.6 &
    #     sleep 5
    #     ./run-hb-atari.sh $env $seed 0.7
    #     wait
    # done
    # npg
    for seed in {0..4}
    do
        ./run-npg-atari.sh $env $seed
    done
done