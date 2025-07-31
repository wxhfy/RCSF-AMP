/home/fyh0106/miniconda3/envs/multi/bin/python -m data_processing.main_preprocess \
 --output_dir ./benchmark1_graph \
 --data_root /home/20T-1/fyh0106/compare/ \
 --benchmark_mode benchmark1 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --esm_model_base_path "/home/fyh0106/work/utils/esm_lora/esm2/" \
 --num_workers 32

 /home/fyh0106/miniconda3/envs/multi/bin/python -m data_processing.main_preprocess \
 --output_dir ./benchmark2_graph \
 --data_root /home/20T-1/fyh0106/compare2/ \
 --benchmark_mode benchmark2 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --esm_model_base_path "/home/fyh0106/work/utils/esm_lora/esm2/" \
 --num_workers 32



