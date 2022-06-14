# Variational autoencoder, 줄여서 VAE는 generative model의 한 종류로, input과 output을 같게 만드는 것을 통해 
# 의미 있는 latent space를 만드는 autoencoder와 비슷하게 encoder와 decoder를 활용해 latent space를 도출하고, 
# 이 latent space로부터 우리가 원하는 output을 decoding함으로써 data generation을 진행한다.
# CNN 모델 기반의 변이형 autoencoder 모델 작성 : MNIST dataset을 사용
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

autoencoder = Sequential()

# encoder : 차원 축소
autoencoder.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))

# decoder : 차원 복구
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')
history = autoencoder.fit(x_train, x_train, epochs = 50, batch_size=128, 
                          validation_data=(x_test, x_test), verbose=2)
print(history.history)

# 학습 결과 출력
random_test = np.random.randint(x_test.shape[0], size = 5)
ae_imgs = autoencoder.predict(x_test)

# 시각화
plt.figure(figsize=(7, 2))

for i, image_idx in enumerate(random_test):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    ax.axis('off')
    ax = plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(ae_imgs[image_idx].reshape(28, 28))
    ax.axis('off')

plt.show()

plt.plot(history.history['loss'], c='r', label='loss')
plt.plot(history.history['val_loss'], c='b', label='val_loss')
plt.legend()
plt.show()
