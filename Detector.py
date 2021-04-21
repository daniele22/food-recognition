import cv2
import json
import torch
import json 
from PIL import Image
import numpy as np 

# import needed libraries for detectors
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
import mmcv.visualization.image as mmcv_image

from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, show_result, init_detector

class Detector:

    def __init__(self):
        # config file name
        self.model_cfg_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        # get weights
        self.model_checkpoint = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

        # build the model from a config file and a checkpoint file
        #self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.model = init_detector(config_file, checkpoint_file, device='cpu')


    def inference(self, img_file):
        result = inference_detector(model, img_file)
        # show the results
        final_img = show_result_pyplot(model, img_file, result, score_thr=0.3, show=False, out_file='result.jpg')

        return result, final_img

