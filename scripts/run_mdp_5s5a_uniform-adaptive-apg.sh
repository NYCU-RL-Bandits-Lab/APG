#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-24 13:23:20
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-11-29 14:46:08
 # @FilePath: /mru/APG/scripts/run_mdp_5s5a_uniform-adaptive-apg.sh
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
env="mdp_5s5a_uniform"
# run
python3 main.py --log_root ./logs/adaptive-apg \
                --fname $env \
                --gamma 0.9 \
                --APG_epoch_size 10000 \
                --APG_adaptive_epoch_size 10000 \
                --APG_graphing_size 10000 \
                --APG_adaptive_graphing_size 10000 \
                --env ./mdp_env/$env.yaml --run_algos APG_adaptive APG
# graph
# python3 graph.py --log_dir ./logs/$env \
#                 --plot_LogLog \
#                 --graphing_size 100000000
# python3 graph.py --log_dir ./logs/$env \
#                 --plot_Value \
#                 --algo APG \
#                 --graphing_size 10000
# python3 graph.py --log_dir ./logs/$env \
#                 --plot_Value \
#                 --algo PG \
#                 --graphing_size 1000000
# python3 graph.py --log_dir ./logs/$env \
#                 --algo APG \
#                 --plot_MomGrad \
#                 --graphing_size 100000