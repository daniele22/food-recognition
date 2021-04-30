# Food Recognition with Deep Learning 

![Food-Challenge](https://i.imgur.com/0G3PEc7.png)

<p align="center">
 <a href="https://discord.gg/GTckBMx"><img src="https://img.shields.io/discord/657211973435392011?style=for-the-badge" alt="chat on Discord"></a>
</p>

# Table of contents
- [🚀 Problem Statement](#-problem-statement)
- [💪 Getting Started](#-getting-started)
  * [Using this repository](#using-this-repository)
  * [Running the code locally](#running-the-code-locally)
- [✨ Web App](#-webapp)
- [🧩 Repository structure](#-repository-structure)
  * [Required files](#required-files)
  * [Other files](#other-files)
- [📎  Links](#-links)
- [✍️ Author](#-author)



# Problem Statement

The goal of this project is to train models which can look at images of food items and detect the individual food items present in them.
AICrowd provides a novel dataset of food images collected using the MyFoodRepo project where numerous volunteer Swiss users provide images of their daily food intake. The images have been hand labelled by a group of experts to map the individual food items to an ontology of Swiss Food items.

![image1](https://i.imgur.com/zS2Nbf0.png)

Therefore:
*   Given Images of Food, we are asked to provide **Instance Segmentation** over the images for the food items.
*   The Training Data is provided in the **COCO format**, making it simpler to load with pre-available COCO data processors in popular libraries.
*   The test set provided in the public dataset is the same to Validation set.

# 💪 Getting Started

The dataset of the [AIcrowd Food Recognition Challenge](https://www.aicrowd.com/challenges/food-recognition-challenge) is available at [https://www.aicrowd.com/challenges/food-recognition-challenge/dataset_files](https://www.aicrowd.com/challenges/food-recognition-challenge/dataset_files) and it's the dataset used in this project.
- We have a total of **24120 RGB images** with **2053 validation**, all in **MS-COCO format** and test set for now is same as validation ( debug mode ). 

## Using this repository
This repository contains the code, the models and checkpoint files needed to train/inference with some algorithms.

It's not necessary to clone the repository, you can just download the notebook file and run it in google colab, this is how the project has been developed. It uses `mmdetection` library for build train and test the models.
MMdetection is an open source object detection toolbox based on PyTorch, with a large Model Zoo with many customised models that can be plugged and tested in with just a single config file modification. You can read more about it at: [mmdetection github](https://github.com/open-mmlab/mmdetection/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniele22/notebbok/FoodRecognition.ipynb)
All the commands for installing the needed dependences are present and executed in the notebook, some examples:

Clone the repository
```bash
git clone https://github.com/daniele22/food-recognition
```

Install dependencies
```bash
pip install -r food.recongition/requirements.txt
pip install mmcv
pip install mmdetection 
...
```

By default the notebook uses Google Drive to load and save file, but you can change teh values of some variables at the beginning of the notebook and set different settings, e.g. download the dataset directly from aicrowd urls.

## Running the code locally

The code has not been tested locally because `mmdetection` requires Linux or MacOs systems, but you can find more info here [MMDetection Getting Started](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md).

# The Web App
This repository contains also code to serve a mmdetection model as a webapp for easy visualization of results. To demo the WebApp you can run same the colab notebook, which will take care of all required dependencies.

Through the web app you can upload pictures and get predictions from AI’s point of view.

Inspired from (https://discourse.aicrowd.com/t/a-flask-webapp-for-maskrcnn-inference-visualization/3984)

# 🧩 Repository structure

## Files

**File** | **Description**
--- | ---
`app.py` | code to run the web app
`detector.py` | contains code to make inference with a model builded with mmdetection
`const.py` | list of classes recognized by the models
`requirements.txt` | List of python packages that should be installed (via `pip`) for run the code

## Folders

**Folder** | **Description**
--- | ---
`models` | contains the models cofig file (for mmdet library), the checkpoints and the training logs results
`static` | this repo will contain the images loaded with the web app
`template` | contains the html file of the web app


# 📎 Some links to useful resources on AICrowd


- 💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/food-recognition-challenge
- 🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/food-recognition-challenge/discussion
- 🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/food-recognition-challenge/leaderboards
- Resources - Round 1
  * [Colab Notebook for Data Analysis and Tutorial](https://colab.research.google.com/drive/1A5p9GX5X3n6OMtLjfhnH6Oeq13tWNtFO#scrollTo=ok54AWT_VoWV)
  * [Baseline with `mmdetection` (pytorch)](https://gitlab.aicrowd.com/nikhil_rayaprolu/food-pytorch-baseline)
  * [Baseline with `matterport-maskrcnn` (keras - tensorflow)](https://gitlab.aicrowd.com/nikhil_rayaprolu/food-recognition)
- Resources - Round 2
  * [Colab Notebook for Data Analysis and Tutorial](https://colab.research.google.com/drive/1vXdv9quZ7CXO5lLCjhyz3jtejRzDq221)
  * [Baseline with `mmdetection` (pytorch)](https://gitlab.aicrowd.com/nikhil_rayaprolu/food-round2)
- Resources - Round 3
  * [Colab Notebook for data exploration](https://discourse.aicrowd.com/t/detectron2-colab-notebook-from-data-exploration-to-training-the-model/3691)
- [Participant contributions](https://discourse.aicrowd.com/tags/c/food-recognition-challenge/112/explainer)
- External resources:
  * [Convert Annotations from MS COCO format to PascalVOC format](https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/anno_coco2voc.py)
  

# ✍️ Author   
**[Daniele Filippini]**
