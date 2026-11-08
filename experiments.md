## claris, person-independent (60 signs, 40 epochs, seeds 0-2)

Command: `python main.py train --dataset claris --model <m> --val-signers <id> --seed <s>`

Train 32319 -> val 61333 (train 832 | val 887):

| Model | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| LSTM | 0.126 | 0.107 | 0.104 | 0.112 | 0.012 |
| GRU  | 0.160 | 0.157 | 0.143 | 0.153 | 0.009 |
| TCN  | 0.268 | 0.228 | 0.256 | 0.251 | 0.021 |

Train 61333 -> val 32319 (train 887 | val 832):

| Model | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| LSTM | 0.081 | 0.108 | 0.108 | 0.099 | 0.016 |
| GRU  | 0.139 | 0.112 | 0.113 | 0.121 | 0.015 |
| TCN  | 0.177 | 0.157 | 0.154 | 0.163 | 0.013 |

Mean over both directions: LSTM 0.106, GRU 0.137, TCN 0.207. Chance level 0.017.

Notes:
- TCN is best in all 6 runs. GRU vs LSTM is not separable in the second direction (seeds 1, 2 differ by 0.004-0.005).
- Result depends on which signer is held out (TCN 0.251 vs 0.163); reason unknown. Needs more signers.
- Best epoch is picked on the validation set itself, so numbers are optimistic.
- Open hypothesis: LSTM/GRU read only the last frame, TCN averages over time.