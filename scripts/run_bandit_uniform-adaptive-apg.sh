#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-15 10:38:24
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-11-29 14:44:54
 # @FilePath: /mru/APG/scripts/run_bandit_uniform-adaptive-apg.sh
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
env="bandit_uniform"
# run
python3 main.py --log_root ./logs/adaptive-apg \
                --fname $env \
                --gamma 0.0 \
                --APG_epoch_size 1000 \
                --APG_adaptive_epoch_size 1000 \
                --APG_graphing_size 1000 \
                --APG_adaptive_graphing_size 1000 \
                --env ./mdp_env/$env.yaml --run_algos APG_adaptive APG
# graph
# python3 graph.py --log_dir ./logs/adaptive-apg/$env \
#                 --plot_LogLog \
#                 --graphing_size 1000
# python3 graph.py --log_dir ./logs/$env \
#                 --algo APG \
#                 --plot_MomGrad \
#                 --graphing_size 20000