from keras.layers import Dense, Activation
from keras import optimizers,regularizers
from keras.models import Sequential
import time
import numpy as np
import cv2
from mnist.loader import MNIST
m = MNIST('./data')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

classes = [0,1,2,3,4,5,6,7,8,9]
x_train,y_train = m.load_training()
x_test,y_test = m.load_testing()
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


n_classes = len(classes)
#0-1 Hot encoding
label_train = np.zeros((y_train.shape[0], n_classes))
a = np.arange(y_train.shape[0], dtype=np.int64)
b = np.array(y_train, dtype=np.int64).reshape((y_train.shape[0],))
label_train[a, b] = 1
label_test = np.zeros((y_test.shape[0], n_classes))
c = np.arange(y_test.shape[0], dtype=np.int64)
d = np.array(y_test, dtype=np.int64).reshape((y_test.shape[0],))
label_test[c, d] = 1



print("Training Data Shape is {}".format(x_train.shape))
print("Training Labels Shape is {}".format(y_train.shape))
print("Testing Data Shape is {}".format(x_test.shape))
print("Testing Labels Shape is {}".format(y_test.shape))


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(10, activation='softmax')) # softmax for probability distribution
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,label_train,epochs = 30 )
val_loss, val_acc = model.evaluate(x_test, label_test)
print(val_loss, val_acc)
print(model.predict(np.reshape(x_test[0],(1,784)),batch_size=1))
model.save('model.h5')