#Import Libraries

import albumentations as A
import os
import cv2
import json
from numpy import partition

# Using albumentations to create a random effect for the images

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2), 
    A.RGBShift(p=0.2), 
    A.VerticalFlip(p=0.5)], 
    bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))


# Create additional 50 images out of 1 images
# Slot them according to respective folders

for folder in ['Train','Test', 'Val']:
    for image in os.listdir(os.path.join('imagewebcam',folder,'images')):
        img = cv2.imread(os.path.join('imagewebcam', folder,'images',image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('imagewebcam', folder, 'labels',f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path,'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]

        try:
            for x in range(50):
                augmented = transform(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) ==0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0

                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        
        except Exception as e:
            print(e)
        

