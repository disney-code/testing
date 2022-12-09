from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from attack import FGSM, PGD
import time
from scipy.io import loadmat
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 


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


# load your model 
model = keras.models.load_model("./saved_model/Lenet5_svhn.h5")

fgsm = FGSM(model, ep=0.01, isRand=True)
pgd = PGD(model, ep=0.01, epochs=10, isRand=True)

# generate adversarial examples at once. 
# advs, labels, fols, ginis = fgsm.generate(x_train, y_train)
# np.savez('./FGSM_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)

# advs, labels, fols, ginis = pgd.generate(x_train, y_train)
# np.savez('./PGD_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)
start_t = time.time()
advs, labels, _, _ = fgsm.generate(x_vali, y_vali)
np.savez('./PGD_FGSM_FINAL_DATA_26Nov/FGSM_Vali.npz', advs=advs, labels=labels)

advs, labels, _, _ = pgd.generate(x_vali, y_vali)
np.savez('./PGD_FGSM_FINAL_DATA_26Nov/PGD_Vali.npz', advs=advs, labels=labels)

advs, labels, _, _ = fgsm.generate(x_test, y_test)
np.savez('./PGD_FGSM_FINAL_DATA_26Nov/FGSM_Test.npz', advs=advs, labels=labels)

advs, labels, _, _ = pgd.generate(x_test, y_test)
np.savez('./PGD_FGSM_FINAL_DATA_26Nov/PGD_Test.npz', advs=advs, labels=labels)

end_time=time.time()
time_taken=end_time-start_t

np.savez("./PGD_FGSM_FINAL_DATA_26Nov/time_taken_to_run_gen_adv_python.npz", time_taken=time_taken)