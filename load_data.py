import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt 
import tensorflow as tf

def get_batch(data_annos, batch_num, data_dir, batch_index, 
                batch_size, data_num, size=(224 ,224), 
                is_agument=False, is_shuffle=False):
    ''' 
        Convert data from image to tensor for training (batch_size, width, height, channel)
        --path_data: chứa data dưới dạng numpy with tuple (path_image, lable)
        --data_dir: path chứa images
        --batch_size: số lượng ảnh trong 1 lần training
        --is_agument: sinh thêm data (images...)
        --is_shuffle: trộn image tăng khả năng học của model
        --size: size ảnh
        --batch_index: số lượng data (len(data)/batch_size)
        ==================================================
        Return: list_image, list_label, list_image_name
    '''
    if batch_index >= batch_num:
        raise ValueError("Batch idx must be in range [0, {}].".format(batch_num - 1))
    
    #Get start and end image index (counting from 0)
    #Chia khoảng và lấy 16 image làm trong 1 lần train 
    start_idx = batch_index * batch_size
    idx_range = []

    for i in range(batch_size):
        idx_range.append((start_idx + i) % data_num)
    
    print('Batch index: {}, counting from 0'.format(batch_index))
    img_tensor = []
    label_list = []
    img_name_list = []
    for i in idx_range:
        #path_name and label
        img_name, label = data_annos[i]
        img = cv2.imread(data_dir + img_name)

        #image agument (nếu is_agument = True)
        #Xử lí thêm phần agument ảnh 

        if size is not None:
            #Padding and resize to out 
            img = cv2.resize(img, tuple(size))
    
        img_tensor.append(img)
        label_list.append(label)
        img_name_list.append(img_name)

    img_tensor = np.array(img_tensor)
    label_list = np.array(label_list)

    return img_tensor, label_list, img_name_list