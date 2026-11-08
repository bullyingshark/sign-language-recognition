## claris, train 32319 -> val 61333 (60 signs, 40 epochs, seeds 0-2)

Command: `python main.py train --dataset claris --model <m> --val-signers 61333 --seed <s>`

| Model | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| LSTM | 0.126 | 0.107 | 0.104 | 0.112 | 0.012 |
| GRU  | 0.160 | 0.157 | 0.143 | 0.153 | 0.009 |
| TCN  | 0.268 | 0.228 | 0.256 | 0.251 | 0.021 |

Notes: chance level 0.017. Best epoch chosen on the same validation set (optimistic).