python -m data_processing.main_preprocess \
 --output_dir ./benchmark1_graph \
 --data_root ./data/benchmark2/ \
 --benchmark_mode benchmark1 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --esm_model_base_path "./data/esm2" \
 --num_workers 32

python -m data_processing.main_preprocess \
 --output_dir ./benchmark2_graph \
 --data_root ./data/benchmark2/ \
 --benchmark_mode benchmark2 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --esm_model_base_path "./data/esm2" \
 --num_workers 32



