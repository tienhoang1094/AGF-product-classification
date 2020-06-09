import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.preprocessing import image
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="normalized/tmpnznlpwgt.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

direction = '/home/peter-linux/Desktop/AGF/Data-collector/product_data/data3/other/'
c = 0
d =os.listdir(direction)
for i in d:
    img_path = direction + i
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    t1=time.time()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    t2=time.time()
    print(t2-t1)
    print(np.argmax(output_data))
    if np.argmax(output_data) != 1:
        c+=1

print(c,len(d))
print(100*(1-c/len(d)),'%')