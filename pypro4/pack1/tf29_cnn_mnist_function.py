# Colab 사용
# CNN : 이미지의 특징을 뽑아 크기를 줄인 후 마지막에 1차원 배열로 만든 후 Dense에 전달하는 방식
# MNIST dataset을 사용
# Function api 방법'''

import tensorflow as tf
from keras import datasets, layers, models
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# CNN은 Channel을 사용하기 때문에 3차원 데이터를 4차원으로 변경해 준다.
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# model : Function api
input_shape = (28, 28, 1) # 입력

img_input = tf.keras.layers.Input(shape=input_shape)

net = layers.Conv2D(filters=16, kernel_size=(3,3), activation = 'relu')(img_input)
net = layers.MaxPooling2D(pool_size = (2,2))(net) #maxpool, maxpooling 상관없다.
net = layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu')(net)
net = layers.MaxPooling2D(pool_size = (2,2))(net)
net = layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu')(net)
net = layers.MaxPooling2D(pool_size = (2,2))(net)

net = layers.Flatten()(net)

net = layers.Dense(units=64, activation='relu')(net)
net = layers.Dropout(0.2)(net)
net = layers.Dense(units=32, activation='relu')(net)
net = layers.Dropout(0.2)(net)

outputs = layers.Dense(units = 10, activation = 'softmax')(net)

model = tf.keras.Model(inputs=img_input, outputs=outputs)

print(model.summary())

# compile
from keras import callbacks
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=3) # 일반적으로 patience=10정도 준다.
history = model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=2, validation_split = 0.2, callbacks=[es])

# 모델 학습 후 모델 저장. 그런데 history 값도 계속 참조하고 싶은 경우 pickle로 저장함
import pickle
history = history.history
with open('his_data.pickle', 'wb') as f:
    pickle.dump(history, f)

print(history)

# 모델 평가 (train / test)
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('train_loss, train_acc:', train_loss, train_acc)
print('test_loss, test_acc:', test_loss, test_acc)

# model save
model.save('tf28.h5')

model2 = tf.keras.models.load_model('tf28.h5')

model2 = tf.keras.models.load_model('tf28.h5')

# predict
import numpy as np
print('예측값:', np.argmax(model2.predict(x_test[:1])))
print('실제값:', y_test[0])

# 시각화
import matplotlib.pyplot as plt
with open('his_data.pickle', 'rb') as f:
    hitsory = pickle.load(f)

def plot_acc():
    plt.plot(history['accuracy'], label='accuracy') 
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title('accuracy') 
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()

plot_acc()
plt.show()

def plot_loss():
    plt.plot(history['loss'], label='loss') 
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('loss') 
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

plot_loss()
plt.show()