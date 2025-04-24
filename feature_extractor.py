import os
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
import rawpy
import cv2

import torch
from torchvision import models, transforms

import imagehash
import subprocess
import json
from datetime import datetime


register_heif_opener()

class feature_extractor:
    def __init__(self):
        os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def load_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 4:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frames.append(frame)
                frames.append(frame)
                frames.append(frame)
        else:
            duration = frame_count / fps

            frame_indices = [int(t * fps) for t in [0, duration * 0.33, duration * 0.66]]
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame at index {frame_idx}/{frame_count}: {video_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            for frame_idx in range(frame_count - 1, -1, -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    break
        cap.release()

        if len(frames) != 4:
            raise ValueError(f"Failed to read 4 frames")
        h, w = frames[0].shape[:2]
        grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        grid[:h, :w] = frames[0]
        grid[:h, w:] = frames[1]
        grid[h:, :w] = frames[2]
        grid[h:, w:] = frames[3]
        return Image.fromarray(grid)

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)
    
    @staticmethod
    def load_raw_image(image_path):
        raw = rawpy.imread(image_path)
        rgb = raw.postprocess()
        return Image.fromarray(rgb)
    
    @staticmethod
    def load_file(file_path):
        ext = file_path.split('.')[-1].lower()
        if ext in ['mp4', 'mov', 'wmv', 'mpg', 'avi', 'mkv']:
            img = feature_extractor.load_video(file_path)
        elif ext in ['rw2']:
            img = feature_extractor.load_raw_image(file_path)
        else:
            img = feature_extractor.load_image(file_path)
            if img.mode == 'RGBA':
                rgb_img = Image.new("RGB", img.size, (0,0,0))
                rgb_img.paste(img, mask=img.split()[3])  # 3번 인덱스는 알파 채널
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        return img
    
    @staticmethod
    def get_datetime(fn):
        result = subprocess.check_output(["exiftool", "-j", fn], universal_newlines=True)
        metadata = json.loads(result)
        dt = None
        if 'DateTimeOriginal' in metadata[0]:
            dt = metadata[0]['DateTimeOriginal']
            if dt[:10] == '0000:00:00':
                dt = None
        if dt is None and 'MediaCreateDate' in metadata[0]:
            dt = metadata[0]['MediaCreateDate']
            if dt[:10] == '0000:00:00':
                dt = None
        if dt is None and 'FileModifyDate' in metadata[0]:
            dt = metadata[0]['FileModifyDate']
            if dt[:10] == '0000:00:00':
                dt = None
        if dt is None:
            mod_time = os.path.getmtime(fn)
            mod_time_dt = datetime.fromtimestamp(mod_time)
            dt = mod_time_dt.strftime("%Y:%m:%d %H:%M:%S")
            print(f'using filesystem modified date: <{dt}> {fn}')
        return dt
        
    @staticmethod
    def get_filesize(fn):
        return os.path.getsize(fn)

    def to_latent(self, img):
        img_tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        latent_vector = torch.flatten(latent_vector, 1)
        return latent_vector.cpu()
    
    def to_features(self, fn):
        img = feature_extractor.load_file(fn)
        return self.to_latent(img), imagehash.average_hash(img), imagehash.dhash(img), feature_extractor.get_datetime(fn), feature_extractor.get_filesize(fn)


valid_exts = [
    'gif',
    'heic',
    'jpeg',
    'jpg',
    'mov',
    'mp4',
    'mpg',
    'png',
    'rw2',
    'wmv'
]
