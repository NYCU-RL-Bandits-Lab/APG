<!--
 * @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @Date: 2023-06-15 13:36:36
 * @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @LastEditTime: 2023-11-01 11:48:03
 * @FilePath: /mru/APG/README.md
 * @Description: 
 * 
-->
# Accelerated Policy Gradient (APG)
**Accelerated Policy Gradient: On the Nesterov Momentum for Reinforcement Learning** [\[arXiv\]](https://arxiv.org/abs/2310.11897)

Yen-Ju Chen, Nai-Chieh Huang, [Ping-Chun Hsieh](https://pinghsieh.github.io/)

*ICML 2023 Workshop on New Frontiers in Learning, Control, and Dynamical Systems*
[\[Paper\]](./assests/APG-ICML-workshop-paper.pdf.pdf), [\[Poster\]](./assests/APG-ICML-workshop-poster.pdf)


<br/><br/>
## Folder Structure
```
.
├── assests/
│   └── APG-ICML-workshop-paper.pdf.pdf
│   └── APG-ICML-workshop-poster.pdf
├── helper/
│   ├── plot.py
│   └── utils.py
├── mdp_env/
│   ├── bandit_hard.yaml
│   ├── bandit_non_monotone.yaml
│   ├── bandit_non_monotone.yaml
│   └── ...
├── scripts/
│   ├── run_bandit_hard.sh
│   ├── run_bandit_non_monotone.sh
│   └── run_bandit_uniform.sh
│   └── ...
├── train/
│   ├── APG.py
│   ├── Bellman.py
│   ├── parameters.py
│   ├── PG_adam.py
│   ├── PG_heavy_ball.py
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
    pip3 install pyyaml termcolor pandas numpy matplotlib tqdm fastparquet
    ```

<br/><br/>
## Quick Start
- Easily run the following code to perform APG & PG on a [testing MDP env](./mdp_env/test.yaml):
    ```py
    python3 main.py --fname test
    ```
    Note: Specify other arguments [here](./train/parameters.py).

- After running `main.py`, one can find the suumary plot in `./logs/test/APG/APG_summary_1000.png`:
    <center class="half">
        <kbd><img src=./logs/test/APG/APG_summary_1000.png></kbd>
    </center>
    
    and `./logs/test/PG/PG_summary_1000.png`:
    <center class="half">
        <kbd><img src=./logs/test/PG/PG_summary_1000.png></kbd>
    </center>

    and `./logs/test/PG_heavy_ball/PG_heavy_ball_summary_1000.png`:
    <center class="half">
        <kbd><img src=./logs/test/PG_heavy_ball/PG_heavy_ball_summary_1000.png></kbd>
    </center>

    and `./logs/test/PG_adam/PG_adam_summary_1000.png`:
    <center class="half">
        <kbd><img src=./logs/test/PG_adam/PG_adam_summary_1000.png></kbd>
    </center>

- Run `graph.py` to get more plot:
    ```py
    python3 graph.py --log_dir ./logs/test \
                     --algo APG \
                     --graphing_size 500 1000 \
                     --plot_Summary \
                     --plot_Value \
                     --plot_LogLog \
                     --plot_MomGrad \
                     --plot_OneStep
    ```
    <center class="half">
        <kbd><img src=./logs/test/APG/APG_value_1000.png width='175'></kbd>
        <kbd><img src=./logs/test/log_log_1000.png width='175'></kbd>
        <kbd><img src=./logs/test/APG/APG_mom_grad_1000.png width='175'></kbd>
        <kbd><img src=./logs/test/APG/APG_one_step_1000.png width='175'></kbd>
    </center>

<br/><br/>
## Random MDP:
- Easily run the following code to perform APG & PG on a `random MDP`:

    ```py
    python3 main.py --random_mdp \
                    --state_action_num 5 5 \
                    --fname test_random_mdp_5s5a 
    ```
    Note: The information of the random MDP will be recorded at [here](./logs/test_random_mdp_5s5a/args.yaml).

<!-- <center class="half">
    <kbd><img src= width='650'></kbd>
</center> -->
    

<br/><br/>
## Reproducing Results
Run the following code to reproduce the numerical results presented in the paper:
- Change mode before running `.sh`:
    ```sh
    chmod +x ./scripts/{file name}.sh
    ```

- Run:
    ```sh
    ./scripts/run_bandit_hard.sh
    ./scripts/run_bandit_non_monotone.sh
    ./scripts/run_bandit_uniform.sh
    ./scripts/run_mdp_5s5a_hard.sh
    ./scripts/run_mdp_5s5a_uniform.sh
    ```

<br/><br/>
## Citation
If you find our repository helpful to your research, please cite our paper:

```
@article{chen2023accelerated,
  title={Accelerated Policy Gradient: On the Nesterov Momentum for Reinforcement Learning},
  author={Chen, Yen-Ju and Huang, Nai-Chieh and Hsieh, Ping-Chun},
  journal={arXiv preprint arXiv:2310.11897},
  year={2023}
}
```