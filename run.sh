#/bin/bash

k_values=(10 50 100)
# lbd_values=(0.001 0.01 0.1 1)
lbd_values=(100 1000 10000)

for k in "${k_values[@]}"; do
    for lbd in "${lbd_values[@]}"; do
        echo "Running with k=$k and lbd=$lbd"
        python3 /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/lp_proj/train.py \
            --dim_feat $k \
            --lbd $lbd \
            --lr 1e-3 \
            --num_iters 5000 \
            --output_name cuda_feat${k}_lbd${lbd}_lr1e-3_5000steps;
    done
done