# may CUDA out of memory if < 35GB
python main.py \
    -t \
    --base configs/stable-diffusion/retail.yaml \
    --gpus 0, \
    --max_epochs 200 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 3 \
    --finetune_from sd-v1-4-full-ema.ckpt