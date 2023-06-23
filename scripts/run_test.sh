#
###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-06-15 10:38:24
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-06-16 11:40:26
 # @FilePath: /mru/APG/scripts/run_test.sh
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
env="test"
# run
python3 main.py --log_root ./logs \
                --fname $env \
                --gamma 0.9 \
                --APG_epoch_size 1000000 \
                --APG_graphing_size 1000000 \
                --PG_epoch_size 1000000 \
                --PG_graphing_size 1000000 \
                --env ./mdp_env/$env.yaml