# GAS-Net, MS COCO: 50 XE epochs + 30 SCST epochs (paper Sec. 4.2)
python train.py --id GASNet_coco --caption_model gasnet --self_critical_after 50 --max_epochs 80 \
    --batch_size 32 --save_checkpoint_every 3540 --num_gpu 1 --gpu_id 0 --beam 0
