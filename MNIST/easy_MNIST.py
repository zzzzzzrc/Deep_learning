from __future__ import print_function
import keras
import csv
from PIL import Image
from keras.datasets import mnist
import input_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
train_data_dir = "D:\\machine_learning\\MNIST\\train\\"
test_data_dir = "D:\\machine_learning\\MNIST\\test\\"
batch_size = 128
num_classes = 10
epochs = 30
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train,y_train = input_data.get_data(train_data_dir)
x_test,y_test = input_data.get_test_data(test_data_dir)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
# (x_test[0].reshape(28,28))*255
# Image.fromarray((x_test[1].reshape(28,28))*255).show()


model = Sequential()
model.add(Conv2D(32,(3,3),
                 activation='relu',
                 input_shape=input_shape
                ))
model.add(Conv2D(64,(3,3) ,activation='relu'
                 ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
#           verbose=1, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_split=0.2)
# score = model.evaluate(x_test, y_test,batch_size=batch_size,verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
output = model.predict_classes(x_test)
output = list(output)
test_label = list(y_test)
predict = np.array([test_label,output])
print(predict[0],predict[0:10])
csvfile = open('D:\\machine_learning\\MNIST\\sample_submission.csv', 'w',newline='')
writer = csv.writer(csvfile)
writer.writerow(['id','label'])
for i in range(28000):
    print (i)
    writer.writerow([predict[0][i],predict[1][i]])
csvfile.close()
print('done')
