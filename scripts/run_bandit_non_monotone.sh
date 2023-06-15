#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-15 10:38:24
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-06-15 13:57:17
 # @FilePath: /mru/APG/scripts/run_bandit_non_monotone.sh
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
# source ./myenv/bin/activate
# param
env="bandit_non_monotone"
# run
python3 main.py --log_root ./logs \
                --fname $env \
                --gamma 0.0 \
                --APG_epoch_size 500 \
                --PG_epoch_size 500 \
                --env ./mdp_env/$env.yaml
# graph
python3 graph.py --log_dir ./logs/$env \
                --algo APG \
                --plot_OneStep \
                --graphing_size 500