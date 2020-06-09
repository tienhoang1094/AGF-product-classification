from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
import numpy as np
import os
import time
import cv2

model = load_model('/home/peter-linux/Desktop/AGF/Data-collector/product_data/test-modeltf2/no-nomal/Epoch93-loss0.014-val0.00414-no-normal.h5')
direction = '/home/peter-linux/Desktop/AGF/Data-collector/product_data/data3/other/'
maping  = ['CCL','part','?','others']
c = 0
d =os.listdir(direction)

for i in d:
    img_path = direction + i
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = x/255
    t1= time.time()
    y = model.predict(x)
    t2= time.time()
    cl = (np.argmax(y))
    # print(y[0][cl])

    print(maping[cl],y[0][cl],i)
    # cvi=cv2.imread(img_path)
    # cv2.imshow('',cvi)
    # cv2.waitKey(0)

#     if np.argmax(y) != 2:
#         c+=1
    print(t2-t1)
# print(c,len(d))
# print(100*(1-c/len(d)),'%')