from PIL import Image
import numpy as np
import tensorflow as tf

target_img = Image.open('/home/minos/문서/project/ANN/single/straight.png')
weight_img = Image.open('/home/minos/문서/project/ANN/weight_v0.png')

print(target_img)
print(weight_img)

list = np.full((30,32), 1/255)

print(list)

target_img = np.multiply(target_img, list)

result = np.multiply(target_img,weight_img)
result_2 = np.multiply(result,weight_img)
result_np = np.expand_dims(result_2, axis=2)

print(result_np.shape)
result_img = tf.keras.preprocessing.image.array_to_img(result_np, data_format='channels_last')

result_img.save('/home/minos/문서/project/ANN/dense_out_straight.png')