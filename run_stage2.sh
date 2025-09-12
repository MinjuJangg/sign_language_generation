
accelerate launch --gpu_ids 1 --main_process_port=29700 --num_processes=1 --use_deepspeed --mixed_precision="fp16"  stage2_train_only_previous.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --image_encoder_g_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K' \
  --json_path='/home/minju/PCDMs1/data/sample_source.json' \
  --image_root_path="/home/minju/PCDMs1/all_test"  \
  --output_dir="/home/minju/PCDMs1/output_source" \
  --img_height=512  \
  --img_width=512   \
  --learning_rate=1e-4 \
  --train_batch_size=4 \
  --max_train_steps=25000 \
  --mixed_precision="fp16" \
  --checkpointing_steps=5000  \
  --noise_offset=0.1 \
  --lr_warmup_steps 5000  \
  --seed 42 \
  --report_to wandb \
  --project_name=PCDMs_Source



