#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-15 10:38:24
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-06-15 13:17:43
 # @FilePath: /mru/tmpppp/scripts/run_bandit_uniform.sh
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
env="bandit_uniform"
# run
python3 main.py --log_root ./logs \
                --fname $env \
                --gamma 0.0 \
                --APG_epoch_size 100000000 \
                --APG_graphing_size 10000000 \
                --PG_epoch_size 100000000 \
                --PG_graphing_size 10000000 \
                --env ./mdp_env/$env.yaml
# graph
python3 graph.py --log_dir ./logs/$env \
                --plot_LogLog \
                --graphing_size 100000000
python3 graph.py --log_dir ./logs/$env \
                --algo APG \
                --plot_MomGrad \
                --graphing_size 20000