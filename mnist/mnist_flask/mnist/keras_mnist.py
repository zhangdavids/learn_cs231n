import numpy as np
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 训练数据集图像是 28 * 28 的格式 标签是0-9的数字

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 像素值介于 0-255 归一化 控制在0-1之内

X_train /= 255
X_test /= 255


def train_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


y_train_ohe = np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([train_y(y_test[i]) for i in range(len(y_test))])


model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# 节点展平
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad',
              metrics=['accuracy'])

model.fit(X_train, y_train_ohe,
          validation_data=(X_test, y_test_ohe),
          epochs=20, batch_size=128)

scores = model.evaluate(X_test, y_test_ohe, verbose=0)




