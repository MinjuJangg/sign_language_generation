import os
import json
import cv2
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# 경로 설정
JSON_PATH = "/SSD/sign/2024_sign_data/preprocessed_json/updated_train.json"
OUTPUT_BASE = "/SSL_NAS/sign_data/kor_sign_generation/all_train"

# 설정 값
CROP_WIDTH = 420
TARGET_SIZE = (512, 512)
SEED = 1234

np.random.seed(SEED)

with open(JSON_PATH, "r") as f:
    test_json = json.load(f)
test_videos = list(test_json.keys())

def crop_resize_image(img, output_path):
    cropped = img.crop((CROP_WIDTH, 0, img.width - CROP_WIDTH, img.height))
    resized = cropped.resize(TARGET_SIZE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resized.save(output_path)
    del cropped, resized
    gc.collect()

@lru_cache(maxsize=100)
def get_json_path(name):
    prefix_map = {
        'CUSH': '쇼핑',
        'CUTO': '관광',
        'LIME': '의료',
        'LICC': '민원 행정'
    }
    for k, folder in prefix_map.items():
        if name.startswith(k):
            return f"/SSD/sign/new_sign/signlan/03.구축데이터/{folder}/{name}.json"

@lru_cache(maxsize=100)
def get_video_path(name):
    prefix_map = {
        'CUSH': '쇼핑',
        'CUTO': '관광',
        'LIME': '의료',
        'LICC': '민원 행정'
    }
    for k, folder in prefix_map.items():
        if name.startswith(k):
            return f"/SSL_NAS/new_sign/signlan/03.구축데이터/{folder}/{name}.mp4"

def save_keypoints(name):
    with open(get_json_path(name), 'r') as f:
        data = json.load(f)
    kpts = data['landmarks']
    save_dir = os.path.join(OUTPUT_BASE, name, "pose")
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(kpts):
        pts = frame['predictions'][0]['keypoints']
        with open(os.path.join(save_dir, f"frame_{i}.txt"), 'w') as f:
            for j in range(0, len(pts), 3):
                x, y = float(pts[j]), float(pts[j+1])
                x = max(0, min(512, (x - 420) / 1080 * 512))
                y = max(0, min(512, (y / 1080) * 512))
                f.write(f"{x} {y}\n")
    gc.collect()
    return len(kpts)

def save_images(name, length):
    cap = cv2.VideoCapture(get_video_path(name))
    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] {name} 프레임 {i} 못 읽음")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        output = os.path.join(OUTPUT_BASE, name, "img", f"frame_{i}.jpg")
        crop_resize_image(pil_img, output)
        del frame, frame_rgb, pil_img
        gc.collect()
    cap.release()


def make_pairs(name, length):
    pair_file = os.path.join(OUTPUT_BASE, name, "annotations.txt")
    with open(pair_file, 'w') as f:
        for i in range(1, length):
            f.write(f"./kor_sign_generation/all_train/{name}/img/frame_0.jpg,./kor_sign_generation/all_train/{name}/img/frame_{i}.jpg\n")

def process_video(name):
    os.makedirs(os.path.join(OUTPUT_BASE, name, "img"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, name, "pose"), exist_ok=True)
    frame_len = save_keypoints(name)
    save_images(name, frame_len)
    make_pairs(name, frame_len)
    gc.collect()

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(process_video, test_videos), total=len(test_videos)))
