from __future__ import print_function
def get_test_data(test_dir):
    x_test = []
    y_test = []

    for item in os.listdir(test_dir):

            img = image.load_img(test_dir+item,target_size=(64, 64))
            x = image.img_to_array(img)
            x_test.append(x)
            y_test.append(item)
    print('There are',len(x_test),'test pictures')
    x_test = np.array(x_test)
    print("x_train has the shape of",x_test.shape)

    return x_test,y_test
import keras
import csv
import os
from keras.preprocessing import image
import numpy as np
import h5py
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam

batch_size = 128
num_classes = 12
epochs = 120


test_data_dir = '/home/work/test/'

x_test,y_test= get_test_data(test_data_dir)

f = h5py.File('/home/work/train_gen_data.h5','r')
f.keys()
x_train= f['data'][:]
y_train = f['label'][:]

f.close()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 input_shape=(64,64,3),
                 name='conv1'
                ))
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2,2),
                       padding='valid',
                       name='maxpool1'))
model.add(BatchNormalization(axis=3,momentum=0.99,epsilon=0.001,name='BN1'))
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 activation='relu',
                 name='conv2'
                 ))
model.add(BatchNormalization(axis=3,momentum=0.99,epsilon=0.001,name='BN2'))

model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2,2),
                       padding='valid',
                       name='maxpool2'))
model.add(Conv2D(64,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 name='conv3'))
model.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='BN3'))
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 name='conv4'))
model.add(BatchNormalization(axis=3,momentum=0.99,epsilon=0.001,name='BN4'))
model.add(MaxPooling2D(name='maxpoll3'))
model.add(Dense(128, activation='relu',name='Dense1'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu',name='Dense2'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',name='Dense3'))


log_filepath = '/home/work/'
model.summary()
adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.00001)

model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])
tb_cb = TensorBoard(log_dir=log_filepath, write_images=True, histogram_freq=1,write_graph=True)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_split=0.1)

output = model.predict_classes(x_test)
output = list(output)
test_label = list(y_test)
predict = np.array([test_label,output])
dic = {'0':'Black-grass',
       '1':'Charlock',
       '2':'Cleavers',
       '3':'Common Chickweed',
       '4':'Common wheat',
       '5':'Fat Hen',
       '6':'Loose Silky-bent',
       '7':'Maize',
       '8':'Scentless Mayweed',
       '9':'Shepherds Purse',
       '10':'Small-flowered Cranesbill',
       '11':'Sugar beet'}
csvfile = open('/home/work/sample_submission.csv', 'w',newline='')
writer = csv.writer(csvfile)
writer.writerow(['file','species'])
for i in range(794):
    print (i)
    writer.writerow([predict[0][i],dic[predict[1][i]]])
csvfile.close()
print('done')
