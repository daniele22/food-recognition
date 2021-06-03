"""
Some examples of data augmentation

@author Daniele Filippini
"""
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A

def visualize(image, title=""):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.title(title)
    plt.show()

image = cv2.imread('./data/train/images/007135.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image, "Original")

transform = A.HorizontalFlip(p=0.5)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "HorizontalFlip")

transform = A.ShiftScaleRotate(p=1, shift_limit=0.0625, scale_limit=0.1, rotate_limit=20)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "ShiftScaleRotate")

transform = A.RandomBrightnessContrast(brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "RandomBrightnessContrast")

transform = A.Blur(blur_limit=25, p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "Blur")

transform = A.MotionBlur(blur_limit=25, p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "MotionBlur")

transform = A.GaussNoise(var_limit=40, p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "GaussNoise")

transform = A.ImageCompression(quality_lower=60, p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "ImageCompression")

transform = A.ChannelShuffle(p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "ChannelShuffle")

transform = A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)
random.seed(7)
augmented_image = transform(image=image)['image']
visualize(augmented_image, "MultiplicativeNoise")


