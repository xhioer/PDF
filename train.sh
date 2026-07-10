export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

mkdir -p /home/outputs/PDF

/data/envs/PDF/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port 12336 \
  clip_finetune.py \
  --dataset prcc \
  --cfg configs/prcc.yaml \
  --root /data/datasets/PRCC \
  --output /home/outputs/PDF
