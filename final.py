#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from turicreate import SFrame
import time
import tensorflow as tf
from skimage import io, transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# In[58]:


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
    
if __name__ == "__main__":
    train_data_dir = '/Users/lexin/DataSet/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
    convert_train_data(train_data_dir)


# In[2]:


def convertjpg(jpgfile, a, outdir,width=40,height=40):
    img=Image.open(jpgfile)
    #try:
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir, a, os.path.basename(jpgfile)))
    #print(os.path.join(outdir, a, os.path.basename(jpgfile)))
    #except Exception as e:
        #print(e)
        
for jpgfile in glob.glob("/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg/*/*.jpg"):
    a = jpgfile.split('/')[-2]
    convertjpg(jpgfile, a, "/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg")


# In[5]:


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 只显示 Error
 
# 读取图片
def read_img(path, w, h):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []

    print('Start read the image ...')
    for index, folder in enumerate(cate):
        # print(index, folder)
        for im in glob.glob(folder + '/*.jpg'):
            # print('Reading The Image: %s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(index)
    print('Finished ...')  
    return np.asarray(imgs, np.float32), np.asarray(labels, np.float32)
 
# 打乱顺序
def messUpOrder(data, label):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    return data, label
 
# 将所有数据分为训练集和验证集
def segmentation(data, label, ratio=0.8):
    num_example = data.shape[0]
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_valid = data[s:]
    y_valid = label[s:]
    return x_train, y_train, x_valid, y_valid


# In[6]:


# 全局变量
imgpath = '/Users/lexin/DataSet/GTSRB_Final_Training_Images_roi_jpg/'
batch_size = 64
nb_classes = 43
epochs = 32
# input image dimensions
img_rows, img_cols = 40, 40
# 卷积滤波器的数量
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
 
#load data
data, label = read_img(path=imgpath, w=img_rows, h=img_cols)
data, label = messUpOrder(data=data, label=label)
X_train, y_train, X_valid, y_valid = segmentation(data=data, label=label)

# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    #X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    #X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 3)

# 类型转换
X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
#X_test /= 255
X_valid /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'valid samples')
 
# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)


# In[7]:


#构建模型
model = Sequential()
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape)) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
model.add(Activation('relu')) #激活层
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(128)) #全连接层1
model.add(Activation('relu')) #激活层
model.add(Dropout(0.5)) #随机失活
model.add(Dense(nb_classes)) #全连接层2
model.add(Activation('softmax')) #Softmax评分

#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#训练模型
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid))


# In[8]:


#评估模型
score = model.evaluate(X_valid, Y_valid, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[10]:


#save architecture
json_string = model.to_json()
open('./model_architecture.json','w').write(json_string)
#save weights
model.save_weights('./model_weights.h5')

