from keras.preprocessing import image
import numpy as np
import os
import h5py
#get train picture and train label
def get_train_data(train_dir):
    x_train = []
    y_train = []
    i = -1
    # for item in os.listdir('D:\\machine_learning\\Plant Seedings CLassification\\train\\'):
    for item in os.listdir(train_dir):
        # i += 1
        # print(i)#label
        # for png in os.listdir(train_dir+item):
            #load picture     picture size 227*227  for alexnet or vgg16
        img = image.load_img(train_dir+item,target_size=(64,64))     #type is img
        x = image.img_to_array(img)                                             #change type to np.ndarray
        x_train.append(x)
        name = item.split('.')
        if name[0] == 'cat':
            y_train.append(0)
        else:
            y_train.append(1)
    print('There are',len(x_train),'pictures')
    x_train = np.array(x_train)                         #type np.ndarray 4D(number,length,width,channel)
    y_train = np.array(y_train)                         #type np.ndarray 1D(number,)
    print("x_train has the shape of",x_train.shape)
    print("y_train has the shape of",y_train.shape)

    return x_train,y_train

def get_test_data(test_dir):
    x_test = []
    y_test = []
    # for item in os.listdir('D:\\machine_learning\\Plant Seedings CLassification\\test\\'):
    for item in os.listdir(test_dir):
            #load picture     picture size 227*227  for alexnet or vgg16
        img = image.load_img(test_dir+item,target_size=(64, 64))     #type is img
        x = image.img_to_array(img)                                   #change type to np.ndarray
        x_test.append(x)
        name = item.split('.')
        y_test.append(name[0])
    print('There are',len(x_test),'test pictures')
    x_test = np.array(x_test)                         #type np.ndarray 4D(number,length,width,channel)                       #type np.ndarray 1D(number,)
    print("x_train has the shape of",x_test.shape)

    return x_test,y_test



#make train h5py file
train_data_dir = 'D:\\machine_learning\\dogs_vs_cats\\train\\'
x_train,y_train = get_train_data(train_data_dir)

#打乱h5文件训练集正负例顺序
index = [i for i in range(len(x_train))]
np.random.shuffle(index)
x_train= x_train[index]
y_train = y_train[index]

f = h5py.File('D:\\machine_learning\\dogs_vs_cats\\train_data.h5','w')#相对路径，绝对路径会报错
f['data']=x_train
f['label']=y_train
f.close()