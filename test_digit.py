import cv2
import numpy as np
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from mnist.loader import MNIST
m = MNIST('./data')

#Load the saved model
model = keras.models.load_model('model.h5')


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


print(x_test[9].shape)
print(y_test[100])
p = np.reshape(x_test[100],(1,784))
predicted = model.predict(p,batch_size=1)
print(predicted)






# while True:
#         _, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         img_array = cv2.bitwise_not(gray)
#         img_size = 28
#         new_array = cv2.resize(img_array, (img_size,img_size))
#         user_test = tf.keras.utils.normalize(new_array, axis = 1)
#         p = np.reshape(user_test,(-1,784))
#         predicted = model.predict(p)
#         print(predicted)

#         cv2.imshow("Capturing", frame)
#         key=cv2.waitKey(1)
#         if key == ord('q'):
#                 break
# video.release()
# cv2.destroyAllWindows()