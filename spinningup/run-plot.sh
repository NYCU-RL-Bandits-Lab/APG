#!/usr/bin/env bash
source /home/m-ru/anaconda3/bin/activate
source activate spinningup
python -m spinup.run plot \
    ./data/Best-BipedalWalker-v3/BipedalWalker-v3-sgd \
    ./data/Best-BipedalWalker-v3/BipedalWalker-v3-npg \
    ./data/Best-BipedalWalker-v3/BipedalWalker-v3-apg \
    --legend \
    "PG" \
    "NPG" \
    "APG (Ours)" \
    --fpath ./images/BipedalWalker-v3.png\
    --smooth 10 --plot_size 20000000 --fig_size 7 5 --font_scale 1.7 --title BipedalWalker-v3