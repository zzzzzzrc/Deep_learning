import os
import numpy as np
from PIL import Image
def get_files(file_dir):
    images = []
    labels = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')         #  将文件名与jpg分隔开
        images.append(file_dir + file)      #  存图像文件名
        labels.append(name[0][-1])          #  name[0]文件名   最后一位为标签
    temp = np.array([images, labels])
    temp = temp.transpose()
    # np.random.shuffle(temp)
    #print(temp)  for  test
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
def get_data(file_dir):
    x,y = get_files(file_dir)
    array_x = []
    array_y = []
    for item in x:
        array = np.array(Image.open(item))
        array_x.append(array)
    for item in y:
        array_y.append(item)
    x_train = np.array(array_x)
    y_train = np.array(array_y)
    return x_train,y_train

def get_test_files(file_dir):
    images = []
    labels = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')         #  将文件名与jpg分隔开
        images.append(file_dir + file)      #  存图像文件名
        labels.append(name[0])          #  name[0]文件名   最后一位为标签
    temp = np.array([images, labels])
    temp = temp.transpose()
    # np.random.shuffle(temp)
    #print(temp)  for  test
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
def get_test_data(file_dir):
    x,y = get_test_files(file_dir)
    array_x = []
    array_y = []
    for item in x:
        array = np.array(Image.open(item))
        array_x.append(array)
    for item in y:
        array_y.append(item)
    x_test = np.array(array_x)
    y_test= np.array(array_y)
    return x_test,y_test





# def get_batch(image,label,image_W, image_H,batch_size,capacity):
#     #转换数据格式
#     image = tf.cast(image, tf.string)
#     label = tf.cast(label, tf.int64)
#     #构造一个队列
#     input_queue = tf.train.slice_input_producer([image, label])
#     label = input_queue[1]
#     image_contents = tf.read_file(input_queue[0])
#     image = tf.image.decode_jpeg(image_contents,channels=1)
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
#     image = tf.image.per_image_standardization(image)
#
#     image_batch, label_batch = tf.train.batch([image, label],
#                                                 batch_size= batch_size,
#                                                 num_threads= 64,
#                                                 capacity = capacity)
#     label_batch = tf.reshape(label_batch, [batch_size])
#     image_batch = tf.cast(image_batch, tf.float32)
#     return image_batch, label_batch
