CUDA_VISIBLE_DEVICES=0 python3 stage2_test_recursive_randomcolor.py \
  --img_weigh 512 \
  --img_height 512 \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --image_encoder_p_path='facebook/dinov2-giant' \
  --img_path='/home/minju/PCDMs/test_data/' \
  --json_path='/home/minju/PCDMs/data/test.json' \
  --save_path="/home/minju/PCDMs/result/gt_cond_recursive" \
  --weights_name="/home/minju/PCDMs/output_randomre/25000" \
  --calculate_metrics

