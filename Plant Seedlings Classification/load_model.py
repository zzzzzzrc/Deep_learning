from keras import models
import input_data
import csv
import numpy as np
model = models.load_model('D:\\machine_learning\\Plant Seedings CLassification\\model\\model.h5')
test_data_dir = 'D:\\machine_learning\\Plant Seedings CLassification\\test\\'
# # the data, shuffled and split between train and test sets
#
# x_train,y_train = input_data.get_train_data(train_data_dir)
x_test,y_test= input_data.get_test_data(test_data_dir)
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
