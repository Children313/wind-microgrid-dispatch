# Wind Power Forecasting: Baseline Comparison

| Rank | Model | Source | Params | MAE (kW) | RMSE (kW) | R² |
|:----:|:------|:------:|-------:|---------:|----------:|---:|
| 1 | PatchTST **🥇** | TSL | 402,701 | 4756.3 | 7423.8 | 0.9475 |
| 2 | DLinear | TSL | 194 | 4782.1 | 7504.5 | 0.9464 |
| 3 | iTransformer | TSL | 410,253 | 4934.7 | 7539.9 | 0.9459 |
| 4 | TimeMixer | TSL | 56,081 | 5252.0 | 7803.9 | 0.9420 |
| 5 | Autoformer | TSL | 277,505 | 11583.6 | 15957.6 | 0.7576 |