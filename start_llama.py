import subprocess
cmd = '/home/mt/miniconda3/envs/py38/bin/python  run_llama.py \
        --device musa \
        --model_name_or_path ./llama_config \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 3 \
        --per_device_eval_batch_size 1 \
        --output_dir ./test-clm \
        --cache_dir ./gpt2_ckpt/wikitext/wikitext-2-raw \
        --block_size 2048 \
        --num_train_epochs 10'.split()
res = subprocess.call(cmd)