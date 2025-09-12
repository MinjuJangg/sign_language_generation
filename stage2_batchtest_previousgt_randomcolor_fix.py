import os
from PIL import Image, ImageStat, ImageEnhance
import numpy as np
from diffusers import UniPCMultistepScheduler
from src.models.stage2_inpaint_unet_2d_condition2 import Stage2_InapintUNet2DConditionModel
from src.pipelines.stage2_inpaint_pipeline_color import Stage2_InpaintDiffusionPipeline
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
import argparse
from transformers import Dinov2Model
from typing import Any, Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as compare_ssim
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import json
import time
import cv2

# def get_contrast_pil(image: Image.Image) -> float:
#     gray_image = image.convert("L")
#     stat = ImageStat.Stat(gray_image)
#     return stat.stddev[0] / 128.0

# def get_hue_saturation_brightness_cv2(image_path: str):
#     img = cv2.imread(image_path)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
#     hue = np.mean(hsv[..., 0])
#     saturation = np.mean(hsv[..., 1])
#     brightness = np.mean(hsv[..., 2])
#     return hue, saturation, brightness

# def sinusoidal_encoding(x: torch.Tensor, dim: int = 64):
#     device = x.device
#     half_dim = dim // 2
#     emb_scale = torch.exp(torch.arange(half_dim, device=device) * (-np.log(10000.0) / (half_dim - 1)))
#     emb = x.unsqueeze(1) * emb_scale.unsqueeze(0)
#     return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# def extract_and_encode_condition(image_path: str, encoding_dim: int = 320):
#     pil_image = Image.open(image_path).convert("RGB")
#     contrast = get_contrast_pil(pil_image)
#     h, s, v = get_hue_saturation_brightness_cv2(image_path)
#     h_norm = h / 180.0
#     s_norm = s / 255.0
#     v_norm = v / 255.0
#     c_norm = contrast
#     cond_vector = torch.tensor([h_norm, s_norm, v_norm, c_norm], dtype=torch.float32).unsqueeze(0)
#     encoded = [sinusoidal_encoding(cond_vector[:, i], dim=encoding_dim) for i in range(4)]
#     return torch.cat(encoded, dim=1).squeeze(0)  # shape: (encoding_dim * 4,)

def get_brightness_pil(image: Image.Image) -> float:
    gray_image = image.convert("L")
    stat = ImageStat.Stat(gray_image)
    brightness = stat.mean[0] / 255.0  # 0~1 범위
    print(brightness)
    normalized = brightness / 0.25  # 밝기 상한 0.2라고 가정
    normalized = min(normalized, 1.0)  # 1을 넘지 않도록 제한
    print(normalized)

    return normalized

def sinusoidal_encoding(x: torch.Tensor, dim: int = 320):
    device = x.device
    half_dim = dim // 2
    emb_scale = torch.exp(torch.arange(half_dim, device=device) * (-np.log(10000.0) / (half_dim - 1)))
    emb = x.unsqueeze(1) * emb_scale.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

def extract_and_encode_brightness_from_image(image_path: str, encoding_dim: int = 1280):
    pil_image = Image.open(image_path).convert("RGB")
    brightness = get_brightness_pil(pil_image)  # 0~1 정규화
    brightness_tensor = torch.tensor([brightness], dtype=torch.float32)
    encoded = sinusoidal_encoding(brightness_tensor, dim=encoding_dim)
    return encoded.squeeze(0)  # shape: (encoding_dim,)

def extract_and_encode_brightness_from_value(brightness_value: float, encoding_dim: int = 1280):
    brightness = brightness_value  # 0~1 범위
    print(brightness)
    brightness_tensor = torch.tensor([brightness], dtype=torch.float32)
    encoded = sinusoidal_encoding(brightness_tensor, dim=encoding_dim)
    return encoded.squeeze(0)

def split_list_into_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



class ImageProjModel_p(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



def inference(args, rank, select_test_datas):

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)


    # save path
    save_dir = "{}/show_guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)
    save_dir_metric = "{}/guidancescale{}_seed{}_numsteps{}/".format(args.save_path, args.guidance_scale, args.seed_number, args.num_inference_steps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


    # model define
    image_proj_model_p_dict = {}
    pose_proj_dict = {}
    unet_dict = {}

    # image_encoder_g = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_g_path).to(device).eval()
    image_encoder_p = Dinov2Model.from_pretrained(args.image_encoder_p_path).to(device).eval()

    image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).eval()
    pose_proj = ControlNetConditioningEmbedding(320, 3, (16, 32, 96, 256)).to(device).eval()

    model_ckpt = "{}/mp_rank_00_model_states.pt".format(args.weights_name)
    model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

    for k in model_sd.keys():
        if k.startswith("pose_proj"):

            pose_proj_dict[k.replace("pose_proj.", "")] = model_sd[k]

        elif k.startswith("image_proj_model_p"):
            image_proj_model_p_dict[k.replace("image_proj_model_p.", "")] = model_sd[k]

        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]

        else:
            print(k)

    pose_proj.load_state_dict(pose_proj_dict)
    image_proj_model_p.load_state_dict(image_proj_model_p_dict)

    pipe = Stage2_InpaintDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,torch_dtype=torch.float16).to(device)

    pipe.unet= Stage2_InapintUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                           in_channels=9, class_embed_type="None",
                                           projection_class_embeddings_input_dim=1024,torch_dtype=torch.float16,
                                           low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(device)

    pipe.unet.load_state_dict(unet_dict)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.enable_xformers_memory_efficient_attention()
    print('====================== json_data: {}, model load finish ==================='.format((args.json_path).split('/')[-1]))


    # number = 0
    all_ssim = []

    start_time = time.time()
    for data in select_test_datas:

        s_img_path = (args.img_path + data["source_image"])
        s_img = Image.open(s_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)

        black_image = Image.new("RGB", s_img.size, (0, 0, 0))
        s_img_t_mask = Image.new("RGB", (s_img.width * 2, s_img.height))
        
        t_img_path = (args.img_path + data["target_image"])
        t_img = Image.open(t_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)

        #brightness_factor = random.uniform(0.8, 5.0)  #(0.8,5.0)
        brightness_factor = 1.0
        t_img = ImageEnhance.Brightness(t_img).enhance(brightness_factor)

        
        p_img_path = (args.img_path + data["prev_image"])
        p_img = Image.open(p_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)
        s_img_t_mask.paste(p_img, (0, 0))
        s_img_t_mask.paste(black_image, (s_img.width, 0))


        s_pose = Image.open(p_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)
        t_pose = Image.open(t_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)
        st_pose = Image.new("RGB", (s_pose.width * 2, s_pose.height))
        st_pose.paste(s_pose, (0, 0))
        st_pose.paste(t_pose, (s_pose.width, 0))

        clip_processor_s_img = clip_image_processor(images=s_img, return_tensors="pt").pixel_values
        s_img_f = image_encoder_p(clip_processor_s_img.to(device)).last_hidden_state
        s_img_proj_f = image_proj_model_p(s_img_f)  # s_img


        vae_image = torch.unsqueeze(img_transform(s_img_t_mask), 0)
        
        color_cond = extract_and_encode_brightness_from_value(0.1).to(device)

        cond_st_pose = torch.unsqueeze(img_transform(st_pose), 0)
        st_pose_f = pose_proj(cond_st_pose.to(device=device))  # t_pose

        # if args.json_path.split('/')[-1].split('_')[0] == "train":
        #     clip_processor_s_img = clip_image_processor(images=t_img, return_tensors="pt").pixel_values
        #     pred_t_img_embed = (image_encoder_g(clip_processor_s_img.to(device)).image_embeds).unsqueeze(1)
        # #
        # elif args.json_path.split('/')[-1].split('_')[0] == "test":
        #     pred_t_img_embed = torch.tensor(np.load(args.target_embed_path + s_img_path.split("/")[-1].replace(".jpg", '_to_') \
        #                                             + t_img_path.split("/")[-1].replace(".jpg", '.npy'))).to(device)
        #     pred_t_img_embed = pred_t_img_embed.unsqueeze(1)
        # else:
        #     raise ValueError("Check the input JSON file path")


        output = pipe(
                height=args.img_height,
                width=args.img_weigh*2,
                guidance_rescale=0.0,
                vae_image=vae_image,
                s_img_proj_f=s_img_proj_f,
                st_pose_f=st_pose_f,
                color_cond=color_cond.unsqueeze(0),
                num_images_per_prompt=4,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            )


        vis_st_pose = Image.new("RGB", (args.img_weigh*2, args.img_height))
        vis_st_pose.paste((Image.open(p_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)), (0, 0))
        vis_st_pose.paste((Image.open(t_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC)), (args.img_weigh, 0))

        #vis_st_pose.paste((Image.p_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC), (0, 0))
        #vis_st_pose.paste((Image.t_img_path.replace("/img/", "/pose_img/").replace(".jpg", "_pose.jpg")).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC), (args.img_weigh, 0))

        # vis_st_image = Image.new("RGB", (args.img_weigh*2, args.img_height))
        # vis_st_image.paste(Image.open(p_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC), (0, 0))
        # vis_st_image.paste(Image.open(t_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC), (args.img_weigh, 0))

        vis_st_image = Image.new("RGB", (args.img_weigh*2, args.img_height))
        vis_st_image.paste(Image.open(p_img_path).convert("RGB").resize((args.img_weigh, args.img_height), Image.BICUBIC), (0, 0))
        vis_st_image.paste(t_img, (args.img_weigh, 0)) 

        if args.calculate_metrics:
            ssim_values = []
            for gen_img in output.images:
                gen_img = gen_img.crop((args.img_weigh,0, args.img_weigh*2,args.img_height))
                ssim_values.append(compare_ssim(np.array(t_img)*255.0, np.array(gen_img)*255.0,
                                                gaussian_weights=True, sigma=1.2,
                                                use_sample_covariance=False, multichannel=True, channel_axis=2,
                                                data_range=(np.array(gen_img)*255.0).max() - (np.array(gen_img)*255.0).min()
                                                ))
            max_value = max(ssim_values)
            all_ssim.append(max_value)
            max_index = ssim_values.index(max_value)
            grid_metric = output.images[max_index].crop((args.img_weigh,0, args.img_weigh*2,args.img_height))
            grid_metric.save(save_dir_metric + s_img_path.split("/")[-1].replace(".jpg", "") + "_to_" + t_img_path.split("/")[-1])
        #else:
            #output.images.insert(0, vis_st_image)
            output.images.insert(0, vis_st_pose)
            output.images.insert(0, vis_st_image)
            grid = image_grid(output.images, 2, 3)
            #grid = image_grid(output.images, 1, len(output.images))  # 한 줄로 배치

            grid.save(
                save_dir + s_img_path.split("/")[-1].replace(".jpg", "") + "_to_" +
                t_img_path.split("/")[-1])

    end_time =time.time()
    print(end_time-start_time)

    if args.calculate_metrics:
        print(sum(all_ssim)/ len(all_ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of an inpaint model of stage2 script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="/mnt/aigc_cq/private/feishen/weights/stable-diffusion-2-1-base",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    #parser.add_argument("--image_encoder_g_path",type=str,default="./OpenCLIP-ViT-H-14",
    #    help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--image_encoder_p_path",type=str,default="./dinov2-giant",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--img_path", type=str,default="./datasets/deepfashing/train_all_png/", help="image path", )
    #parser.add_argument("--pose_path", type=str,default="./datasets/deepfashing/openpose_all_img/",help="pose path", )
    parser.add_argument("--json_path", type=str,default="./datasets/deepfashing/test_data.json",help="json path", )
    #parser.add_argument("--target_embed_path", type=str,default="./save_data/stage1/guidancescale0_seed42_numsteps20/",help="t_img_embed path", )
    parser.add_argument("--save_path", type=str, default="./save_data/stage2", help="save path", )
    parser.add_argument("--guidance_scale",type=int,default=2.0,help="guidance_scale",)
    parser.add_argument("--seed_number",type=int,default=42,help="seed number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="num_inference_steps",)
    parser.add_argument("--img_weigh",type=int,default=512,help="image weigh",)
    parser.add_argument("--img_height",type=int,default=512,help="image height",)
    parser.add_argument("--calculate_metrics",  action='store_true', help="caculate ssim", )
    parser.add_argument("--weights_name", type=str, default="./Checkpoints/stage2_checkpoints/512",help="weights number", )
    args = parser.parse_args()
    print(args)

    num_devices = torch.cuda.device_count()
    print("using {} num_processes inference".format(num_devices))

    test_data = json.load(open(args.json_path))

    select_test_datas = test_data
    print(len(select_test_datas))

    mp.set_start_method("spawn")
    data_list = split_list_into_chunks(select_test_datas, num_devices)


    processes = []
    for rank in range(num_devices):
        p = mp.Process(target=inference, args=(args, rank, data_list[rank] ))
        processes.append(p)
        p.start()

    for rank, p in enumerate(processes):
        p.join()



