# Colab 사용
# CNN : 이미지의 특징을 뽑아 크기를 줄인 후 마지막에 1차원 배열로 만든 후 Dense에 전달하는 방식
# MNIST dataset을 사용
# subclassing model api 방법
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras import datasets, Model

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

train_images = x_train.reshape((60000, 28, 28, 1))
train_images = train_images / 255.0
test_images = x_test.reshape((10000, 28, 28, 1))
test_images = test_images / 255.0
train_labels = y_train
test_labels = y_test

# tf.data.Dataset.from_tensor_slices() : 입력 파이프라인을 가능하게 함. 데이터를  골고루 섞기 등의 작업이 가능
# Dataset.from_tensor_slices() 경험해 보기 -------------
import numpy as np
x = np.random.sample((5, 2))
print(x)
dset =  tf.data.Dataset.from_tensor_slices(x)
print(dset)
dset =  tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(2)
print(dset)
for a in dset:
    print(a)
    
# MNIST의 학습 데이터(60000, 28, 28)가 입력되면 6만 개의 slice를 만들고 각 slice는 28 * 28 형태로 만들기
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(28)
print(train_ds)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(28) # test는 shuffle 할 필요가없다.
print(test_ds)

# subclassing model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=[3,3], padding='valid', activation='relu')
        self.pool1 = MaxPool2D((2,2))
        self.conv2 = Conv2D(filters=32, kernel_size=[3,3], padding='valid', activation='relu')
        self.pool2 = MaxPool2D((2,2))
        self.flatten = Flatten(dtype='float32')
        self.d1 = Dense(32, activation = 'relu')
        self.drop1 = Dropout(rate=0.2)
        self.d2 = Dense(10, activation = 'softmax')

    def call(self, inputs):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.flatten(net)
        net = self.d1(net)
        net = self.drop1(net)
        net = self.d2(net)
        return net

model = MyModel()    
temp_inputs = tf.keras.Input(shape=(28,28,1))
model(temp_inputs)

from keras import callbacks
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=5, verbose=2,
          use_multiprocessing=True, workers=2) #use_multiprocessing=False, workers=1 // use_multiprocessing=True, workers=2 (멀티프로세싱가능)
score = model.evaluate(test_images, test_labels)
print('loss:',score[0])
print('acc:',score[1])