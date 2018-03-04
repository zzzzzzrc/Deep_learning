from keras.preprocessing import image
import numpy as np
import os
import h5py


def get_train_data(train_dir):
    x_train = []
    y_train = []
    i = -1

    for item in os.listdir(train_dir):
        i += 1
        print(i)
        for png in os.listdir(train_dir+item):

            img = image.load_img(train_dir+item+'/'+png,target_size=(299,299))
            x = image.img_to_array(img)
            x_train.append(x)
            y_train.append(i)
    print('There are',len(x_train),'pictures')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("x_train has the shape of",x_train.shape)
    print("y_train has the shape of",y_train.shape)

    return x_train,y_train


train_data_dir = '/home/work/train_gen/'

x_train,y_train = get_train_data(train_data_dir)

index = [i for i in range(len(x_train))]
np.random.shuffle(index)
x_train= x_train[index]
y_train = y_train[index]

f = h5py.File('/home/work/train_gen_data_299.h5','w')
f['data']=x_train
f['label']=y_train
f.close()
