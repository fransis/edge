import os
import pickle
import numpy as np
from glob import glob
from tqdm.notebook import tqdm

from PIL import Image
from pillow_heif import register_heif_opener
import imagehash
import cv2

import torch
import torch.nn.functional as F
from torchvision import models, transforms


register_heif_opener()

class feature_extractor:
    def __init__(self, device='cuda'):
        os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
        self.device = device
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.to(self.device)

    def load_movie(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        timestamps = [0, duration * 0.33, duration * 0.66, duration]
        frames = []
        for t in timestamps:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count-1, frame_idx))
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at timestamp {t} seconds")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        h, w = frames[0].shape[:2]
        grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        grid[:h, :w] = frames[0]
        grid[:h, w:] = frames[1]
        grid[h:, :w] = frames[2]
        grid[h:, w:] = frames[3]
        return Image.fromarray(grid)
    
    def load_image(self, image_path):
        return Image.open(image_path)
    
    def load_file(self, file_path):
        if file_path.split('.')[-1].lower() in ['mp4', 'mov']:
            return self.load_movie(file_path)
        else:
            img = self.load_image(file_path)
            if img.mode == 'RGBA':
                rgb_img = Image.new("RGB", img.size, (0,0,0))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    
    def to_latent(self, img):
        img_tensor = self.preprocess(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        latent_vector = torch.flatten(latent_vector, 1)
        return latent_vector.cpu()
    
    def to_dhash(self, img):
        return imagehash.dhash(img)
    
    def to_ahash(self, img):
        return imagehash.average_hash(img)
    
    def to_features(self, fn):
        img = self.load_file(fn)
        return self.to_latent(img), self.to_ahash(img), self.to_dhash(img)


class duplicated_remover:
    def __init__(self, base_path, device='cuda'):
        self.device = device
        self.base_path = base_path
        self.extractor = feature_extractor(device)
        self.base_features = self.make_dict_features(base_path, 'base_features.pkl')
        self.src_features = {}

    def remove(self, src_path, name, by='latent', delete=True, ulimit=0.999, llimit=0.9):
        self.src_features = self.make_dict_features(src_path, f'{name}.pkl')
        if by == 'latent':
            simularity_calculator = duplicated_remover.calc_simularity_by_latend
            feature_idx = 0
        elif by == 'ahash':
            simularity_calculator = duplicated_remover.calc_simularity_by_hash
            feature_idx = 1
        elif by == 'dhash':
            simularity_calculator = duplicated_remover.calc_simularity_by_hash
            feature_idx = 2
        else:
            print(f'unknown method: {by}')
            return
        for sfn, (latent, ahash, dhash) in self.src_features.items():
            if os.path.exists(sfn):
                bfn, min_score = self.calc_minimum_simularity(sfn, feature_idx, simularity_calculator)
                print(f'{fn} -> {bfn}: socre={min_score}')
    
    def calc_minimum_simularity_by_latent(self, sfn, feature_idx, simularity_calculator):
        sfeatures = self.src_features[sfn]
        min_fn = ''
        min_score = np.inf
        for bfn, bfeatures in self.base_features.items():
            score = simularity_calculator(sfeatures[feature_idx], bfeatures[feature_idx])
            if score < min_score:
                min_score = score
                min_fn = bfn
        return min_fn, min_score

    @staticmethod
    def calc_simularity_by_hash(hash1, hash2):
        return hash1 - hash2
    
    @staticmethod
    def calc_simularity_by_latend(latent1, latent2):
        return F.cosine_similarity(latent1, latent2, dim=1)

    def make_dict_features(self, path, dict_fname):
        if os.path.exists(dict_fname):
            with open(dict_fname, 'rb') as f:
                dict_features = pickle.load(f)
        else:
            dict_features = {}
        cnt = 0
        for fn in tqdm(glob(base_path + '/**/*.*', recursive=True)):
            fn = fn.replace('\\', '/')
            if fn in dict_features:
                continue
            ext = fn.split('.')[-1].lower()
            if ext not in ['gif', 'heic', 'jpeg', 'jpg', 'mov', 'mp4', 'png']:
                print(f'Unknown ext: {ext}')
            else:
                dict_features[fn] = self.extractor.to_features(fn)
                cnt += 1
                if cnt % 100 == 0:
                    with open(dict_fname, 'wb') as f:
                        pickle.dump(dict_features, f)
        with open(dict_fname, 'wb') as f:
            pickle.dump(dict_features, f)
        return dict_features


if __name__=='__main__':
    src_path = 'D:/data/Photo_backup/fransis.jhlee@gmail.com - Google Photos'
    base_path = 'D:/data/Photo_backup/yearly'

    exts = set()
    for fn in tqdm(glob(base_path + '/**/*.*', recursive=True)):
        exts.add(fn.split('.')[-1].lower())
    for fn in tqdm(glob(src_path + '/**/*.*', recursive=True)):
        exts.add(fn.split('.')[-1].lower())  
    print(list(exts))
  
    dup_remover = duplicated_remover(base_path)
    dup_remover.remove(src_path, 'src_features', 'dhash', False)

  
    dup_remover = duplicated_remover(base_path)
