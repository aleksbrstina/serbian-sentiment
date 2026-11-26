# BiLSTM experiments

_Generated from MLflow by `scripts/summarize_experiments.py` — do not edit by hand._

| Run | lr | dropout | embedding_dropout | pooling | truncation | max_len | batch_size | hidden_size | Val macro F1 | Best epoch | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-lr1e-3_drop0.5 | 0.001 | 0.5 | 0.2 | max_mean | head_tail | 256 | 16 | 64 | 0.7515 | 6 | 200 |
| exp-lr2e-3_drop0.3 | 0.002 | 0.3 | 0.1 | max_mean | head_tail | 256 | 16 | 64 | **0.7632** | 5 | 177 |
| exp-lr1e-3_drop0.5_maxpool | 0.001 | 0.5 | 0.2 | max | head_tail | 256 | 16 | 64 | 0.7626 | 18 | 356 |
| exp-lr1e-3_drop0.5_head | 0.001 | 0.5 | 0.2 | max_mean | head | 256 | 16 | 64 | 0.6853 | 5 | 152 |
