#keras图片预处理

# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='wrap',
            rescale=2)
def img_generate (filedir):
    for item in os.listdir(filedir):
        print(item)
        for file in os.listdir(filedir+item):
            # print(file)
# img_generate('D:\\machine_learning\\Plant Seedings CLassification\\train\\')
            img = load_img(filedir+item+'\\'+file,target_size=(256,256))
            x = img_to_array(img)  # shape (256, 256, 3)
            x = x.reshape((1,) + x.shape)  # shape (1, 256, 256, 3)
            i = 0
            for batch in datagen.flow(x,
                                        batch_size=1,
                                       save_to_dir=filedir+item+'\\',#生成后的图像保存路径
                                       save_prefix='img_gen',
                                       save_format='png'):
                 i += 1
                 if i > 10:
                     break
img_generate('D:\\machine_learning\\Plant Seedings CLassification\\train\\')
