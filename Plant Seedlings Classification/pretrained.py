import numpy as np # linear algebra

import os, cv2
import input_data
import h5py
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications import *
#split validation data
split=False
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
if split:
    import random
    import shutil
    # make dir and mv
    os.mkdir('D:\\machine_learning\\Plant Seedings CLassification\\dev\\')
    for category in CATEGORIES:
        os.mkdir('D:\\machine_learning\\Plant Seedings CLassification\\dev\\' + category)
        name = os.listdir('D:\\machine_learning\\Plant Seedings CLassification\\train\\train\\' + category)
        random.shuffle(name)
        todev = name[:int(len(name) * .2)]
        for file in todev:
            shutil.move(os.path.join('D:\\machine_learning\\Plant Seedings CLassification\\train\\train', category, file), os.path.join('D:\\machine_learning\\Plant Seedings CLassification\\dev', category))
dim = 224
epochs = 300
learning_rate = 0.0001
batch_size = 16

callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0),
              # ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0),
              ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

base_model = ResNet50(input_shape=(dim, dim, 3), include_top=False, weights=None,pooling='avg') # Average pooling reduces output dimensions
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = Dense(256, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
# Load any existing weights
# if os.path.isfile(weights):
#     model.load_weights(weights)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
val_data = ImageDataGenerator(
    rescale=1. / 255)
val_gen = val_data.flow_from_directory(
    'D:\\machine_learning\\Plant Seedings CLassification\\dev\\',
    target_size=(224,224),
    batch_size = 16,
    class_mode='categorical'
)

# model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

# ------ TRAINING ------
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_split=0.1,callbacks=callbacks)
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
#                     steps_per_epoch=len(x_train), epochs=epochs,verbose=1,callbacks=callbacks,validation_data=val_data)
train_generator = train_datagen.flow_from_directory(
    'D:\\machine_learning\\Plant Seedings CLassification\\train\\train',
    target_size=(224, 224  ),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)
model.fit_generator(train_generator,
        steps_per_epoch=250,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks,
        validation_steps=50)
# ------ TESTING ------
'''for f, species in tqdm(df_test.values, miniters=100):
    img = cv2.imread('test/{}'.format(f))
    x_test.append(cv2.resize(img, (dim, dim)))

x_test = np.array(x_test, np.float32)
print(x_test.shape)

if os.path.isfile(weights):
    model.load_weights(weights)

p_test = model.predict(x_test, verbose=1)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])

df_test['species'] = preds
df_test.to_csv('submission.csv', index=False)'''

#-------------------------------------------------------------------------------------
# train_data_dir = 'D:\\machine_learning\\Plant Seedings CLassification\\train_gen\\'
'''test_data_dir = 'D:\\machine_learning\\Plant Seedings CLassification\\test\\'
# # the data, shuffled and split between train and test sets
#
# x_train,y_train = input_data.get_train_data(train_data_dir)
x_test,y_test= input_data.get_test_data(test_data_dir)

output = model.predict(x_test)
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
print('done')'''
