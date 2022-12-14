from tensorflow import keras
import tensorflow as tf
import numpy as np
from models import Lenet5
from scipy.io import loadmat
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
data_train=loadmat("train_32x32.mat")      
x_train=data_train['X']
y_train=data_train['y']%10
x_train=x_train.transpose(3,0,1,2)
x_train=x_train/255.0
y_train=keras.utils.to_categorical(y_train, 10)

x_vali=x_train[60000:]
x_train=x_train[:60000]
y_vali=y_train[60000:]
y_train=y_train[:60000]

data_test=loadmat("test_32x32.mat")
x_test=data_test['X']
y_test=data_test['y']%10
x_test=x_test.transpose(3,0,1,2)
x_test=x_test/255.0
y_test=keras.utils.to_categorical(y_test, 10)

# def load_svhn(path=None):
#     f = np.load(path)
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#     f.close()

#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.

#     y_train = keras.utils.to_categorical(y_train, 10)
#     y_test = keras.utils.to_categorical(y_test, 10)
    
#     return x_train, x_test, y_train, y_test


# path = "./svhn_grey.npz"
# x_train, x_test, y_train, y_test = load_svhn(path)


lenet5 = Lenet5()
lenet5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lenet5.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_vali,y_vali))

lenet5.evaluate(x_test, y_test)

lenet5.save("./Lenet5_svhn.h5")

print("x_train shape: ", x_train.shape)
print("x_vali shape: ", x_vali.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_vali shape: ", y_vali.shape)
print("y_test shape: ", y_test.shape)



