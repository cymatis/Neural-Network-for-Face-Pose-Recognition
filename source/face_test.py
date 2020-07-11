# incompatable with tf < 2.0.0b or use tf-nightly (2.2.0 causes None shape error)
import tensorflow as tf 

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

####################################################

base_dir = os.getcwd()
test_dir = os.path.join(base_dir, 'Face dataset_png/test')
class_name_list = os.listdir(os.path.join(base_dir, 'Face dataset/test'))

img_count = []
file_count_dir = []
IMG_HEIGHT = 30
IMG_WIDTH = 32

for n in range(len(class_name_list)):
    file_count_dir.append(os.listdir(os.path.join(test_dir, class_name_list[n])))
    img_count.append(len(file_count_dir[n]))

test_image_generator = ImageDataGenerator(rescale=1./255) # rescale image byte from 0-255 to 0-1

# make tuple of image and label for test, (image, label)
test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode = 'grayscale',
                                                           class_mode='categorical')

####################################################

model_dir = os.path.join(base_dir, 'models')
model = load_model(model_dir)

model.summary()

####################################################

result_raw = model.predict(test_data_gen, verbose=2)

print(result_raw)

print("Lable list:", class_name_list)
print("")

####################################################

k = 0 # for counting current label number
m = 0 # for first prediction count exception
correct = 0 # for counting correct anwsers

####################################################

for n in range(len(result_raw)): # iterate the number of result from prediction
    result_id = np.argmax(result_raw[n]) # from nth prediction, takes the highest number from output prediction score
    result_label = class_name_list[result_id] # match the actual label from prediction score.
    
    if m == img_count[k]: # kth label, showing anwser from the start
        print("Answer #", class_name_list[k])
        print("") 
        k = k + 1 # tracking current label number
        m = 0 # for first prediction count exception

    if result_label == class_name_list[k]: # if prediction corrects, 1+ to correct
        correct = correct + 1

    acc = round((correct/(n+1))*100,1) # calculate acc
    print("Subject #", n+1," //  Label:",class_name_list[k],"  //  Prediction:", result_label) # showing every prediction result 

    if n == (len(result_raw)-1): # for telling last image of current label
        print("Answer #", class_name_list[k])
        print("")

    m = m + 1 # k번 라벨 m번 이미지

####################################################

print("Total number of image :", len(result_raw))
print("Correct :",correct,"// Wrong :",(len(result_raw)-correct))
print("Test Accuracy :", acc,"""%""")