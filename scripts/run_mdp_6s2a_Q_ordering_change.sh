#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-28 14:52:52
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-06-28 23:16:36
 # @FilePath: /mru/APG/scripts/run_mdp_6s2a_Q_ordering_change.sh
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
env="mdp_6s2a_Q_ordering_change"
# run                
python3 main.py --log_root ./logs \
                --fname $env \
                --gamma 0.9 \
                --run_algos APG \
                --APG_epoch_size 100000 \
                --env ./mdp_env/$env.yaml
# graph
python3 graph.py --log_dir ./logs/$env \
                 --graphing_size 20000 \
                 --plot_Q