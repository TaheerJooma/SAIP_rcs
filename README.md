# <ins> SAIP_rcs <ins>

## Setup:
1. Create an environment based on Python 3.11 using the [Requirements.txt](Requirements.txt) file.

## Overview:
* `rc.py` contains the reservoir computing model.
* `error.py` contains basic functions for MAE, MSE, and RMSE.
* `one_step_and_res_pred.py`, compares the one-step and reservoir prediction of our model to the actual observed time-series.
* `vpts.py`, calculates the valid prediction time of our model.
* `corr_map_recon.py`, plots and compares the actual vs predicted correlation maps.
* `fp_pred.py`, finds and benchmarks the actual vs predicted fixed points.
* `lyp_pred.py`, finds and benchmarks the actual vs predicted Lyapunov exponents.
