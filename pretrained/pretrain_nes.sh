#!/bin/bash

modes=("CCE" "CE");
max_actions=(12 16)

for mode in "${modes[@]}"; do
    for ma in $max_actions; do
        python nes_main.py --batch_size=64\
            --checkpoint_dir="pretrained/${mode}_${ma}"\
            --num_strategies=$ma,$ma\
            --random_games="L2_INVARIANT,L1_INVARIANT,LINF_INVARIANT,EMPIRICAL_DISC_GAME"\
            --iterations=100000\
            --log_every=10000\
            --eval_every=10000\
            --max_actions=$ma\
            --objective="EPS_MRE"\
            --welfare_coeff=0.0 --entropy_coeff=10.0 --epsilon_max=1\
            --max_grad_norm=1.0\
            --learning_rate=0.0001\
            --mode=$mode\
            --dual_channels=64\
            --save_for_deb=true\
            --use_checkpoints=true\
            --seed=42
    done
done
