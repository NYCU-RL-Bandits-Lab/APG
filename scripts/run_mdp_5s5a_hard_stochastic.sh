#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-15 10:38:24
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-10-31 23:32:06
 # @FilePath: /mru/APG/scripts/run_mdp_5s5a_hard_stochastic.sh
 # @Description: 
 # 
### 
# change path if needed
LIST=`ls $search_dir`
if echo ${LIST[*]} | grep -qw "main.py"; then
    echo "found"
else
    echo "cd .."
    cd ..
    LIST=`ls $search_dir`
    if echo ${LIST[*]} | grep -qw "main.py"; then
        echo "found"
    else
        echo "not found"
        exit 1
    fi
fi
# activate env
source ./myenv/bin/activate
# param
# env="bandit_hard"
env="mdp_5s5a_hard"
num_epoch=1000000
# run
python3 main.py --log_root ./logs \
                --fname $env-stochastic \
                --gamma 0.9 \
                --APG_epoch_size $num_epoch \
                --APG_graphing_size $num_epoch \
                --PG_epoch_size $num_epoch \
                --PG_graphing_size $num_epoch \
                --PG_adam_epoch_size $num_epoch \
                --PG_adam_graphing_size $num_epoch \
                --PG_heavy_ball_epoch_size $num_epoch \
                --PG_heavy_ball_graphing_size $num_epoch \
                --env ./mdp_env/$env.yaml \
                --run_algos PG APG \
                --stochastic --seed_num 50
#
# graph
# python3 graph.py --log_dir ./logs/$env-stochastic \
#                 --plot_Summary \
#                 --graphing_size 10000 \
#                 --algo PG
# python3 graph.py --log_dir ./logs/$env-stochastic \
#                 --plot_Summary \
#                 --graphing_size 10000 \
#                 --algo APG
python3 graph.py --log_dir ./logs/$env-stochastic \
                --plot_LogLog \
                --graphing_size $num_epoch