#!/usr/bin/env python
# coding: utf-8

# In[160]:


import turicreate as tc
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import glob
import skimage.io as io
from skimage import data_dir,transform,io,color
from skimage import io,data,color
import PIL.Image


# In[161]:


def convert_train_data(file_dir):

    # 这是图片转换成jpg后另存为的根目录，在运行程序前需要自己先创建
    root_dir = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg'
 
    directories = [file for file in os.listdir(file_dir)  if os.path.isdir(os.path.join(file_dir, file))]
    # print(directories)
 
    for files in directories:
        path = os.path.join(root_dir,files)
        # 判断path路径是否存在，不存在就先创建路径
        if not os.path.exists(path):
            os.makedirs(path)
        # print(path)
 
        data_dir = os.path.join(file_dir, files)
        # print(data_dir)

        # file_name里面每个元素都是以.ppm为后缀的文件的绝对地址
        file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir)  if f.endswith(".ppm")]
 
        for f in os.listdir(data_dir):
            # 获取注解文件的绝对地址
            if f.endswith(".csv"):
                csv_dir = os.path.join(data_dir, f)
                # print(csv_dir)

        # csv_data是一个DataFrama形式的数据结构
        csv_data = pd.read_csv(csv_dir)
 
        csv_data_array = np.array(csv_data)
        # print(csv_data_array)
 
        for i in range(csv_data_array.shape[0]):
            csv_data_list = np.array(csv_data)[i,:].tolist()[0].split(";")
            # print(csv_data_list)

            # 获取该data_dir目录下每张图片的绝对地址
            sample_dir = os.path.join(data_dir, csv_data_list[0])
            # print(sample_dir)

            # 获取兴趣ROI区域
            img = PIL.Image.open(sample_dir)
            box = (int(csv_data_list[3]),int(csv_data_list[4]),int(csv_data_list[5]),int(csv_data_list[6]))
            roi_img = img.crop(box)

            # 截取到兴趣区域后，准备另存为的地址
            new_dir = os.path.join(path, csv_data_list[0].split(".")[0] + ".jpg")
            # print(new_dir)
 
            roi_img.save(new_dir, 'JPEG')


# In[162]:


if __name__ == "__main__":
    train_data_dir = '/Users/lexin/DataSet/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
    convert_train_data(train_data_dir)


# In[165]:


def convert_gray(f, **args):
    rgb=io.imread(f)   
    gray=color.rgb2gray(rgb)     
    dst=transform.resize(gray,(40,40))       
    return dst 

data_dir = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg'

list_name = []
for root, dirs, files in os.walk(data_dir):
    list_name.append(root.split('/')[-1])
list_name.remove(list_name[0])

list_img = []
for root, dirs, files in os.walk(data_dir):
    list_img.append(files)
list_img.remove(list_img[0])

for j in range(len(list_name)):
    str = data_dir+'/'+list_name[j]+'/*.jpg'
    coll = io.ImageCollection(str,load_func=convert_gray)
    createdir = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg_gray/'+list_name[j]+'/'
    os.makedirs(createdir)
    for i in range(len(coll)):
        io.imsave(createdir+list_img[j][i]+'.jpg',coll[i])


# In[166]:


img_folder = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg_gray'
data = tc.image_analysis.load_images(img_folder, with_path=True)
data


# In[167]:


def aa(path, list_name):
    for i in range(len(list_name)):
        if list_name[i] in path:
            return list_name[i]
            
path = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg_gray'
list_name = []

for root, dirs, files in os.walk(path):
    list_name.append(root.split('/')[-1])
list_name.remove(list_name[0])

data['label'] = data['path'].apply(lambda path: aa(path, list_name))
data


# In[168]:


data.explore()


# In[169]:


train_data, test_data = data.random_split(0.8, seed=2)


# In[170]:


model = tc.image_classifier.create(train_data, target='label')


# In[171]:


predictions = model.predict(test_data)


# In[172]:


metrics = model.evaluate(test_data)
print(metrics['accuracy'])

