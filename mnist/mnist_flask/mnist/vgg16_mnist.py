from keras.applications.vgg16 import VGG16

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD

from keras.datasets import mnist

import cv2
import h5py as h5py
import numpy as np

model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in model_vgg.layers:
    layer.trainable = False
model = Flatten(name="flatten")(model_vgg.output)
model = Dense(10, activation='softmax')(model)
model_vgg_mnist = Model(model_vgg.input, model, name='vgg16')

model_vgg_mnist.summary()

# 有1496万个网络权重 本地机器很大可能不能把整个模型和数据放入内存进行训练 建议使用ec2

# 224 可以 降低为 112 甚至是 56 28

sgd = SGD(lr=0.05, decay=1e-5)
model_vgg_mnist.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# opencv 把图像从32*32 变成 224*224 把黑白图像 装换为 rgb 图像 并且把训练数据转化为张量形式
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')


def tran(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


y_train_ohe = np.array([tran(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([tran(y_test[i] for i in range(len(y_test)))])

model_vgg_mnist.fit(X_train, y_train_ohe,
                    validation_data=(X_test, y_test_ohe),
                    epochs=200, batch_size=128)
