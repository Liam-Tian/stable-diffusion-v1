# test on pre-trained model
#python scripts/txt2img.py \
#     --prompt "a photograph of an astronaut riding a horse" \
#     --plms \
#     --H 512 --W 512 \
#     --n_samples 4 \
#     --outdir 'outputs/pre-trained-samples' \
#     --ckpt 'sd-v1-4-full-ema.ckpt'

#### test on retail ####
python scripts/txt2img.py \
    --prompt '0.00' \
    --outdir 'outputs/generated_retail' \
    --H 512 --W 512 \
    --n_samples 4 \
    --config 'configs/stable-diffusion/retail.yaml' \
    --ckpt 'rotation.ckpt'