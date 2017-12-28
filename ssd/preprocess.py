import copy
import numpy as np
import random
from random import shuffle
import cv2
from keras.applications.imagenet_utils import preprocess_input

class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        for i in range(self.crop_attempts):
            random_scale = np.random.random()
            random_scale *= (self.crop_area_range[1] -
                             self.crop_area_range[0])
            random_scale += self.crop_area_range[0]
            target_area = random_scale * img_area
            random_ratio = np.random.random()
            random_ratio *= (self.aspect_ratio_range[1] -
                             self.aspect_ratio_range[0])
            random_ratio += self.aspect_ratio_range[0]
            w = np.round(np.sqrt(target_area * random_ratio))     
            h = np.round(np.sqrt(target_area / random_ratio))            
            if np.random.random() < 0.5:
                w, h = h, w
            w = min(w, img_w)
            w_rel = w / img_w
            w = int(w)
            h = min(h, img_w)
            h_rel = h / img_h
            h = int(h)
            x = np.random.random() * (img_w - w)
            x_rel = x / img_w
            x = int(x)
            y = np.random.random() * (img_h - h)
            y_rel = y / img_h
            y = int(y)
            img = img[y:y+h, x:x+w]
            new_targets = []
            for box in targets:
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                    xmin = (box[0] - x) / w_rel
                    ymin = (box[1] - y) / h_rel
                    xmax = (box[2] - x) / w_rel
                    ymax = (box[3] - y) / h_rel
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(1, xmax)
                    ymax = min(1, ymax)
                    box[:4] = [xmin, ymin, xmax, ymax]
                    new_targets.append(box)
            new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
            return img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            output_keys = []
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = cv2.imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = cv2.resize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                output_keys.append(key)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_keys = copy.copy(output_keys)
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    output_keys = []
                    inputs = []
                    targets = []
                    
                    # preprocess input: preprocess_input(tmp_inp, mode="tf")
                    tmp_inp /=127.5
                    tmp_inp -= 1.
                    
                    yield tmp_keys, tmp_inp, tmp_targets
                    
class SeqGenerator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size, seq_len):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.seq_len = seq_len
    
    def generate(self, train=True):
        while True:
            if train:
                keys = self.train_keys
            else:
                keys = self.val_keys
            output_keys = []
            inputs = []
            targets = []
            
            seq_len = self.seq_len
            image_size = self.image_size
            batch_size = self.batch_size
            batch_key_ids = random.sample([i for i in range(len(keys)-seq_len)], batch_size)
            
            batch_data = []
            batch_target = []
            for batch_key_id in batch_key_ids:
                seq_data = np.zeros([seq_len,]+list(image_size))
                for i in range(seq_len):
                    key = keys[batch_key_id+i]
                    img_path = self.path_prefix + key
                    img = cv2.imread(img_path).astype('float32')
                    img = cv2.resize(img, (self.image_size[0], self.image_size[1])).astype('float32')
                    seq_data[i,:,:,:] = img
                batch_data.append(seq_data)
                y = self.gt[keys[batch_key_id+seq_len]].copy()
                y = self.bbox_util.assign_boxes(y)
                batch_target.append(y)
            
            tmp_inp = np.array(batch_data)
            tmp_targets = np.array(batch_target)
            batch_data = []
            batch_target = []
            
            # preprocess input: preprocess_input(tmp_inp, mode="tf")
            tmp_inp /=127.5
            tmp_inp -= 1.
            
            yield tmp_inp, tmp_targets
            
            
            
            