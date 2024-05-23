# True Gradient MDP

## Folder Structure
```
.
├── helper/
│   ├── plot.py
│   └── utils.py
├── mdp_env/
│   ├── bandit_hard.yaml
│   ├── bandit_non_monotone.yaml
│   └── ...
├── scripts/
│   ├── run_all.sh
│   ├── run_bandit_hard.sh
│   ├── run_bandit_non_monotone.sh
│   └── run_bandit_uniform.sh
│   └── ...
├── train/
│   ├── APG_adaptive.py
│   ├── APG.py
│   ├── Bellman.py
│   ├── parameters.py
│   ├── NPAG.py
│   ├── PG_heavy_ball.py
│   ├── PG.py
│   ├── PI.py
│   └── Saver.py
├── main.py
├── plot.ipynb
├── Readme.md
└── requirements.txt
```
Note: Add `<>.yaml` in the directory [./mdp_env](./mdp_env) if you want to test other MDP / bandit setting.

<br/><br/>
## Environment
- Python 3.8.5
    ```sh
    pip3 install -r requirements.txt
    ```
    or
    ```sh
    pip3 install pyyaml pandas numpy matplotlib fastparquet loguru seaborn
    ```

<br/><br/>
## Quick Start
- Easily run the following code to perform APG on a [MDP with 5 states & 5 actions](./mdp_env/mdp_5s5a_uniform.yaml):
    ```py
    python3 main.py --fname test --algo APG
    ```
    Note: Specify other arguments [here](./train/parameters.py).

- After running `main.py`, one can find the suumary plot in `./logs/test/APG_summary_2000.png`:
    <center class="half">
        <kbd><img src=./logs/test/APG_summary_2000.png></kbd>
    </center>

- Find more plot in [./plot.ipynb](./plot.ipynb)   

<br/><br/>
## Random MDP:
- Easily run the following code to perform APG & PG on a `random MDP`:

    ```py
    python3 main.py --random_mdp \
                    --state_action_num 5 5 \
                    --fname test_random_mdp_5s5a 
    ```
    Note: The information of the random MDP will be recorded at [here](./logs/test_random_mdp_5s5a/APG/args.yaml).

    

<br/><br/>
## Reproducing Results
Run the following code to reproduce the numerical results presented in the paper:
- Change mode before running `.sh`:
    ```sh
    chmod +x ./scripts/{file name}.sh
    ```

- Run:
    ```sh
    ./scripts/run_all.sh
    ```

- Run [./plot.ipynb](./plot.ipynb)