# Auto Encoder 모델 : 주어진 데이터를 학습하여 데이터 분포를 따르는 유사한 데이터를 생성하는 모델
# 비지도 학습
# Encoder로 차원을 축소 후 Decoder로 차원을 확대하여 노이즈가 제거된 비슷한 유형의 결과 생성

# 간단한 autoencoder 연습 : MNIST dataset 사용

import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model

encoding_dim = 32

# network 설계
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
#autoencoder
autoencoder = Model(input_img, decoded)
print(autoencoder.summary())

# encoder
encoder = Model(input_img, encoded)
# decoder
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_layer(encoded_input))
print(decoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# mnist data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

x_train_flatten = x_train.reshape((x_train.shape[0], -1))
x_test_flatten = x_test.reshape((x_test.shape[0], -1))
print(x_train_flatten.shape) #(60000, 784)
print(x_test_flatten.shape) #(10000, 784)

autoencoder.fit(x_train_flatten, x_train_flatten, batch_size=256, epochs=50,
                validation_data=(x_test_flatten, x_test_flatten))

encoded_imgs = encoder.predict(x_test_flatten)
decoded_imgs = decoder.predict(encoded_imgs)

# 시각화
import matplotlib.pyplot as plt

n=10
plt.figure(figsize=(10, 2))

for i in range(1, n+1):
    # 원본 데이터
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_flatten[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 재 구성된 데이터
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()