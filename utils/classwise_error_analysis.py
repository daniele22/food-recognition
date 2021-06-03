"""
Script used to make analysis of classwise results of HTC model.

@author Daniele Filippini
"""
import pandas as pd
import numpy as np
import json

from pprint import pprint # For beautiful print!
import os
import sys
import random
import pickle

# For reading annotations file
from pycocotools.coco import COCO
from pycocotools import coco, cocoeval, _mask
from pycocotools import mask as maskUtils 
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

# For data visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
matplotlib.use('TkAgg')

# Constants of the file
EVAL_FILE_PATH = 'content/drive/MyDrive/ML/models/htc_x101/classwise_eval_htc_x101.txt'
ERROR_TYPE_FILE_PATH = 'content/drive/MyDrive/ML/models/htc_x101/classwise_error_types.txt'

CSV_EVAL_FILE_PATH = "content/drive/MyDrive/ML/models/htc_x101/classwise_eval.csv"
CSV_ERROR_TYPE_FILE_PATH = "content/drive/MyDrive/ML/models/htc_x101/classwise_error_types.csv"


with open("./dataset/val/annotations.json") as f:
  val_annotations_data = json.load(f)

val_coco = COCO("./dataset/val/annotations.json")
train_coco = COCO("./dataset/train/annotations.json")

# # df = pd.read_csv('classwise_eval2.txt', sep="|", header=None)
# # print(df)

###################################################################################
# TO READ EVAL FILE
###################################################################################
def create_csv_classwise_eval_file():
  """
  This function reads the txt file obtained from mmdetection and transforms it to a cvs file
  """
  with open(EVAL_FILE_PATH) as f:
    lines = f.readlines()

  food_list = []
  map_list = []
  for i, line in enumerate(lines):
    l = line.split()
    if len(l) < 13:
        print("skip line ", i)
        print("line:", l)
    else:
        food_list.append(l[1])
        map_list.append(l[3])
        food_list.append(l[5])
        map_list.append(l[7])
        food_list.append(l[9])
        map_list.append(l[11])

  print("len", len(food_list))
  print("len", len(map_list))

  d = { 'category': food_list, 'ap': map_list}

  df = pd.DataFrame(d)
  print(df)

  df.to_csv(CSV_EVAL_FILE_PATH, index=False, )

###################################################################################
# TO READ ERROR TYPES FILE
###################################################################################
def create_csv_classwise_error_file():
  """
  This function reads the txt file obtained from mmdetection and transforms it to a cvs file
  """
  with open(ERROR_TYPE_FILE_PATH) as f:
    lines = f.readlines()

  food_list = []
  cls_list = []
  loc_list = []
  both_list = []
  dupe_list = []
  bkg_list = []
  miss_list = []
  for i, line in enumerate(lines):
    l = line.split()
    print(l)
    if len(l) < 7:
        print("skip line ", i)
        print("line:", l)
    else:
        food_list.append(l[0])
        cls_list.append(l[2])
        loc_list.append(l[3])
        both_list.append(l[4])
        dupe_list.append(l[5])
        bkg_list.append(l[6])
        miss_list.append(l[7])

  print("len", len(food_list))
  print("len", len(cls_list))

  d = { 'category': food_list, 'cls': cls_list, 'loc': loc_list, 'both': both_list, 'dupe': dupe_list, 'bkg': bkg_list, 'miss': miss_list}

  df = pd.DataFrame(d)
  print(df)

  df.to_csv(CSV_ERROR_TYPE_FILE_PATH, index=False, )

############################### check difference #######################################################

def check_difference(path1, path2):
  """
  Check the difference between two dataframes
  """
  df1 = pd.read_csv(path1)
  print(df1)

  df2 = pd.read_csv(path2)
  print(df2)

  missing_values = set(df1.iloc[:, 0]).symmetric_difference(set(df2.iloc[:, 0]))

  print("missing val:")
  print(missing_values)

# Read eval dataframe
df_eval = pd.read_csv(CSV_EVAL_FILE_PATH)
print("Dataframe evaluation:")
print(df_eval)
print(df_eval.info())

def check_null(df):
  """
  checks if some rows have nunll values
  """
  is_NaN = df.isnull()
  row_has_NaN = is_NaN.any(axis=1)
  rows_with_NaN = df[row_has_NaN]
  return rows_with_NaN

rows_with_NaN = check_null(df_eval)
print(rows_with_NaN)
print(rows_with_NaN['category'])

# for cat in rows_with_NaN['category']:
#     #print("CAT:", cat)
#     catIds = val_coco.getCatIds(catNms=[cat])
#     print(cat, " - id:", catIds)
#     catId = catIds[0]
#     annIds = val_coco.getAnnIds(catIds=[catId])  # get annotations ids from image ids list
#     print("anns:", annIds)
#     print("Lenght", len(annIds))
#     anns = val_coco.loadAnns(annIds)  # get the entire annotations needed

# fill null values with 0.000
df_eval = df_eval.fillna(value=0.000)

# sns.barplot(x='category', y='ap', data=df)
# plt.show()

# sort values 
df_eval = df_eval.sort_values(by='ap', ascending=False)

print(df_eval)

best_items = df_eval.head(20)
worst_items = df_eval.tail(20)

def plot_items(df, y_val, title="", bar_color='b'):
  g = sns.barplot(x='category', y=y_val, data=df, color=bar_color)
  g.set_xticklabels(g.get_xticklabels(), rotation=90)
  plt.title(title)
  plt.show()

print("Best items:")
print(best_items)
plot_items(best_items, 'ap', title="Foods with higher AP", bar_color='b')
best_items_names = best_items.iloc[:, 0].values
print(best_items_names)

print("Worst items:")
print(worst_items)
plot_items(worst_items, 'ap', title="Foods with smaller AP", bar_color='r')
worst_items_names = worst_items.iloc[:, 0].values


category_info = train_coco.loadCats(train_coco.getCatIds())
category_names = [_["name"] for _ in category_info]
category_names_readable = [_["name_readable"] for _ in category_info]
no_categories = len(category_names)
# Getting all categoriy with respective to their total images
def getImgsInfos(coco_data):
  """
  function that return all the informations relative to the imgs in a pandas dataframe
  :param coco_data: data in COCO format using pycocotools library function
  :return {no_images_per_category, img_info} where no_images_per_category is a object of type {cat_name: no_of_img}
  and img_info is the pandas df containing the images annotation info
  """
  no_images_per_category = {}
  tot_img = 0
  for n, i in enumerate(coco_data.getCatIds()):
    imgIds = coco_data.getImgIds(catIds=i)
    label = category_names[n]
    no_images_per_category[label] = len(imgIds)
    tot_img = tot_img + len(imgIds)
  # build dataframe with the images informations
  img_info = pd.DataFrame(coco_data.loadImgs(coco_data.getImgIds()))

  # show results
  # print("Total = ", tot_img) # the total correspond to the number of annotations not of the images
  # print("Number of images per category:")
  # pprint(no_images_per_category)

  return no_images_per_category, img_info

# training set info
no_images_per_category_train, img_info_train = getImgsInfos(train_coco)

# # number of elements for the different categories
# for item in worst_items['category']:
#   print("ITEM in worst:", item, no_images_per_category_train[item])

# for item in best_items['category']:
#   print("ITEM in best:", item, no_images_per_category_train[item])

# Read erro file
df_err = pd.read_csv(CSV_ERROR_TYPE_FILE_PATH)

def sort_and_plot(df, y_val):
  df = df.sort_values(by=y_val, ascending=False)
  # print("DF sorted")
  # print(df)
  worst_items = df.head(20)
  print(worst_items)
  plot_items(worst_items, y_val)

# sort_and_plot(df_err, 'cls')
# sort_and_plot(df_err, 'loc')
# sort_and_plot(df_err, 'both')
# sort_and_plot(df_err, 'dupe')
# sort_and_plot(df_err, 'bkg')
# sort_and_plot(df_err, 'miss')

# sorting a dict
img_per_class_sorted = sorted(no_images_per_category_train.items(), key=lambda item: item[1], reverse=True)

print("sorted array")
pprint(img_per_class_sorted)
class_with_most_img = [img_per_class_sorted[i][0] for i in range(0, 20)]
class_with_less_img = [img_per_class_sorted[i][0] for i in range((len(img_per_class_sorted)-20-1), len(img_per_class_sorted)-1)]
print("newlist", class_with_most_img)
print("newlist2", class_with_less_img)
#print(img_per_class_sorted.keys())

def get_sub_df(df, class_name_list):
  df = df.loc[df['category'].isin(class_name_list)]
  print(df)
  return df

df1 = get_sub_df(df_err, worst_items_names)
print("Elements with worst AP analysis")
print(df1)
plot_items(df1, 'cls', title="Classfication error for categories with smallest AP", bar_color='b')
plot_items(df1, 'miss', title="Miss GT error for categories with smallest AP", bar_color='r')

df2 = get_sub_df(df_err, best_items_names)
print("Elements with worst AP analysis")
print(df2)

# dd = df_err.loc[df_err['category'].isin(class_with_most_img)]
# print(dd)

# dd = df_eval.loc[df_eval['category'].isin(class_with_most_img)]
# print(dd)

# dd2 = df_err.loc[df_err['category'].isin(class_with_less_img)]
# print(dd2)

# dd = df_eval.loc[df_eval['category'].isin(class_with_less_img)]
# print(dd)


# Average precision division in bins and count number of classes per bin
sns.histplot(data=df_eval, x="ap", bins=25)
plt.title("AP per class distribution")
plt.show()

# Plot with the different types of errors distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
sns.histplot(data=df_err, x="cls", bins=25, ax=axes[0, 0], color='b')
axes[0, 0].set_title("Classification Error distribution")
sns.histplot(data=df_err, x="loc", bins=25, ax=axes[0, 1], color='g')
axes[0 ,1].set_title("Location Error distribution")
sns.histplot(data=df_err, x="both", bins=25, ax=axes[0, 2], color='m')
axes[0, 2].set_title("Both Cls and Loc Error distribution")
sns.histplot(data=df_err, x="dupe", bins=25, ax=axes[1, 0], color='y')
axes[1, 0].set_title("Duplicate Error distribution")
sns.histplot(data=df_err, x="bkg", bins=25, ax=axes[1, 1], color='c')
axes[1, 1].set_title("Background Error distribution")
sns.histplot(data=df_err, x="miss", bins=25, ax=axes[1, 2], color='r')
axes[1, 2].set_title("Miss GT Error distribution")
plt.show()