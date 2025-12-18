#!/bin/bash
python benchmark/run_benchmark.py

python benchmark/pyatf_mttkrp.py
python benchmark/pyatf_spmv.py
python benchmark/pyatf_harris.py

python benchmark/opentuner_mttkrp.py
python benchmark/opentuner_spmv.py
python benchmark/opentuner_harris.py

python plot_catbench.py --input-dir ./catbench_results 