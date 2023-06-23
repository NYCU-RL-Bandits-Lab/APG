<!--
 * @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @Date: 2023-06-15 13:36:36
 * @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @LastEditTime: 2023-06-21 13:24:15
 * @FilePath: /mru/APG/README.md
 * @Description: 
 * 
-->
# APG
**Target: Could Nesterov’s accelerated gradient reach the convergence rate of $O(1/t^2)$ ?**

<br/><br/>
## Folder structure
```
.
├── helper/
│   ├── plot.py
│   └── utils.py
├── mdp_env/
│   ├── bandit_hard.yaml
│   ├── bandit_non_monotone.yaml
│   └── bandit_uniform.yaml
├── scripts/
│   ├── run_bandit_hard.sh
│   ├── run_bandit_non_monotone.sh
│   └── run_bandit_uniform.sh
├── train/
│   ├── APG.py
│   ├── Bellman.py
│   ├── parameters.py
│   ├── PG.py
│   ├── PI.py
│   └── Saver.py
├── .gitignore
├── graph.py
├── LICENSE
├── main.py
├── Readme.md
└── requirements.txt
```
Note: Add `.yaml` in the directory `./mdp_env` if you want to test other MDP / bandit setting.

<br/><br/>
## Environment
- Python 3.8.5
    ```sh
    pip3 install -r requirements.txt
    ```
    or
    ```sh
    pip3 install pyyaml termcolor pandas numpy matplotlib tqdm fastparquet mpmath
    ```
    
<br/><br/>
## Run code
- change mode before running `.sh`:
    ```sh
    chmod +x ./scripts/{file name}.sh
    ```

- Run:
    ```sh
    ./scripts/run_bandit_hard.sh
    ./scripts/run_bandit_non_monotone.sh
    ./scripts/run_bandit_uniform.sh
    ```

<!-- - If you want to plot more graph...
    ```sh
    cd ./run_sh
    ./plot.sh
    ```
    Note: make sure to modified the file path in `plot.sh` first. -->

<br/><br/>
<!-- ## e.g.
- After executing `run_{environments name}.sh`, one can find the plot below in: `./simple_MDP/{env name}/{date}/{algorithm}_plot/{algorithm}_train_stats_{epoch}.png`</br><br/>
- see example in `./simple_MDP/bandit_uniform_3a/PG_APG_20230228-111539/APG_plot/`

<center class="half">
    <kbd><img src=./simple_MDP/bandit_uniform_3a/PG_APG_20230228-111539/APG_plot/APG_train_stats_1000.png width='650'></kbd>
</center></br><br/>

- If more specific plot is needed, one can run `plot.sh` after modifying the file path in `plot.sh`
- see example in `./simple_MDP/bandit_uniform_3a/PG_APG_20230228-111539/APG_plot/`
- see example in `./simple_MDP/hard_init_2s_3a/PG_APG_20230218-014234/APG_plot/`

<center class="half">
    <kbd><img src=./simple_MDP/bandit_uniform_3a/PG_APG_20230228-111539/APG_plot/APG_grad_vs_norm_5000.png width='650'></kbd>
</center></br><br/>

<center class="half">
    <kbd><img src=./simple_MDP/hard_init_2s_3a/PG_APG_20230218-014234/APG_plot/APG_delta_theta_10000.png width='650'></kbd>
</center></br><br/>

<center class="half">
    <kbd><img src=./simple_MDP/hard_init_2s_3a/PG_APG_20230218-014234/APG_plot/APG_pi_10000.png width='650'></kbd>
</center></br><br/>

<center class="half">
    <kbd><img src=./simple_MDP/hard_init_2s_3a/PG_APG_20230218-014234/APG_plot/APG_pi_10000.png width='650'></kbd>
</center></br><br/> -->
