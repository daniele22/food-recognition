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
from mmdet.apis import train_detector, inference_detector, show_result_pyplot, init_detector
from mmdet.models.detectors import BaseDetector
from mmcv import Config

class Detector:

    def __init__(self):
        # config file name
        self.model_cfg_file = "/content/food-recognition/models/mask_rcnn_r50/mask_rcnn_r50.py"
        # get weights
        self.model_checkpoint = '/content/drive/MyDrive/ML/mask_rcnn_r50/epoch_8.pth'
        #self.cfg = Config.fromfile("/content/food-recognition/models/mask_rcnn_r50/mask_rcnn_r50.py")
        # build the model from a config file and a checkpoint file
        #self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.model = init_detector(self.model_cfg_file, self.model_checkpoint, device='cpu')


    def inference(self, filepath, prediction_path, score_thr=0.3):
        print("Inference detection on:", filepath, " - prediction in:", prediction_path)
        result = inference_detector(self.model, filepath)
        # show the results
        #final_img = show_result_pyplot(self.model, filepath, result, score_thr=0.3, title='Detection result', wait_time=0)

        if hasattr(self.model, 'module'):
            mymodel = self.model.module
        else:
            mymodel = self.model
        mymodel.show_result(
            img=filepath,
            result=result,
            score_thr=score_thr,
            show=False,
            wait_time=0,
            win_name="Detection result",
            bbox_color=(255, 0, 0),
            mask_color=(200, 150, 0),
            text_color=(255, 255, 255),  # white
            out_file=prediction_path  # save results on a specific file
        )
        print("End detection")
        return result

