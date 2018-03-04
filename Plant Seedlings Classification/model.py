from __future__ import print_function
import keras
import csv
import input_data
import numpy as np
import h5py
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
batch_size = 128
num_classes = 12
epochs = 300

# train_data_dir = 'D:\\machine_learning\\Plant Seedings CLassification\\train_gen\\'
test_data_dir = 'D:\\machine_learning\\Plant Seedings CLassification\\test\\'
# # the data, shuffled and split between train and test sets
#
# x_train,y_train = input_data.get_train_data(train_data_dir)
# x_test,y_test= input_data.get_test_data(test_data_dir)
#
# 打乱h5文件训练集正负例顺序
# index = [i for i in range(len(x_train))]
# np.random.shuffle(index)
# x_train= x_train[index]
# y_train = y_train[index]
#
# f = h5py.File('D:\\machine_learning\\Plant Seedings CLassification\\train_gen_data_71.h5','w')#相对路径，绝对路径会报错
# f['data']=x_train
# f['label']=y_train
# f.close()
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 3, 227, 227)
#     x_test = x_test.reshape(x_test.shape[0], 3, 227, 227)
#     input_shape = (3, 227, 227)
# else:
#     x_train = x_train.reshape(x_train.shape[0], 227, 227, 3)
#     x_test = x_test.reshape(x_test.shape[0], 227, 227, 3)
#     input_shape = (227, 227, 3)
# f = h5py.File('D:\\machine_learning\\Plant Seedings CLassification\\train_gen_data_71.h5','r')   #打开h5文件
# f.keys()                            #可以查看所有的主键
# x_train= f['data'][:]                    #取出主键为data的所有的键值
# y_train = f['label'][:]
#
# f.close()


# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# # convert class vectors to binary class matrices
# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
# (x_test[0].reshape(28,28))*255
# Image.fromarray((x_test[1].reshape(28,28))*255).show()
# plt.imshow(x_train[0])
# plt.show()
# print(y_train[0])
model = Sequential()
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 input_shape=(71,71,3),
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
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 name='conv3'))
model.add(BatchNormalization(axis=3,momentum=0.99,epsilon=0.001,name='BN3'))
model.add(Conv2D(64,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2,2),
                       padding='valid',
                       name='maxpool3'))
model.add(BatchNormalization(axis=3,momentum=0.99,epsilon=0.001,name='BN4'))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='Dense1'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu',name='Dense2'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',name='Dense3'))


log_filepath = 'D:\\machine_learning\\Plant Seedings CLassification\\log\\'
model.summary()
adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
# sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])
tb_cb = TensorBoard(log_dir=log_filepath, write_images=True, histogram_freq=1,write_graph=True)
checkpoint = ModelCheckpoint(filepath='D:\\machine_learning\\Plant Seedings CLassification\\model\\model.h5',save_best_only=True)
lr = ReduceLROnPlateau(factor=0.75,patience=3)
earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
# 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的
#权值，每层输出值的分布直方图
# history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
# verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
#           verbose=1, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_split=0.1,callbacks=[tb_cb,checkpoint,lr,earlystop])
# score = model.evaluate(x_test, y_test,batch_size=batch_size,verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
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
csvfile = open('D:\\machine_learning\\Plant Seedings CLassification\\sample_submission.csv', 'w',newline='')
writer = csv.writer(csvfile)
writer.writerow(['file','species'])
for i in range(794):
    print (i)
    writer.writerow([predict[0][i],dic[predict[1][i]]])
csvfile.close()
print('done')
