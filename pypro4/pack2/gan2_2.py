# GAN : 칼라 이미지를 사용
import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channel = 3

# 생성자
generator_input = keras.Input(shape=(latent_dim, ))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# 합성곱 층 추가
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 이미지를 업샘플링
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# 합성곱 층 추가
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channel, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)  
print(generator.summary())

# 판별자
discriminator_input = keras.Input(shape=(height, width, channel))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)  
print(discriminator.summary())# 판별자
discriminator_input = keras.Input(shape=(height, width, channel))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)  
print(discriminator.summary())

from keras.optimizers import Adam, RMSprop, SGD
opti = RMSprop(learning_rate=0.0008, clipvalue=1.0, decay=1e-8)  # clipvalue : 기울기 값 자르기. 기울기 폭주 방지
discriminator.compile(optimizer=opti, loss='binary_crossentropy')

# 적대적 네트워크 형성
discriminator.trainable = False

gan_input = keras.Input(shape = (latent_dim, ))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

opti = RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=opti, loss='binary_crossentropy')

print(gan.summary())

""" DCGAN 훈련 방법 : 매 반복마다 다음을 수행한다:
1. 잠재 공간에서 무작위로 포인트를 뽑는다(랜덤 노이즈).
2. 이 랜덤 노이즈를 사용해 `generator`에서 이미지를 생성.
3. 생성된 이미지와 진짜 이미지를 섞음.
4. 진짜와 가짜가 섞인 이미지와 이에 대응하는 타깃을 사용해 `discriminator`를 훈련한다.
타깃은 “진짜"(실제 이미지일 경우) 또는 “가짜"(생성된 이미지일 경우)다.
5. 잠재 공간에서 무작위로 새로운 포인트를 뽑는다.
6. 이 랜덤 벡터를 사용해 `gan`을 훈련한다. 모든 타깃은 “진짜"로 설정한다.
판별자가 생성된 이미지를 모두 “진짜 이미지"라고 예측하도록 생성자의 가중치를 업데이트한다
(`gan` 안에서 판별자는 동결되기 때문에 생성자만 업데이트). 생성자는 판별자를 속이도록 훈련한다.
"""

# 훈련
import os
from keras.preprocessing import image
from keras.datasets import cifar10
from keras.utils import array_to_img
(x_train, y_train), (_, _) = cifar10.load_data()

x_train = x_train[y_train.flatten() == 6]  # 개구리 이미지 선택
# print(x_train[0], ' ', y_train[0])

x_train = x_train.reshape((x_train.shape[0],) + (height, width, channel)).astype('float32') / 255.

iterations = 1001
batch_size = 20
save_dir = "./gan_images/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
start = 0
for step in range(1, iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)  # 가짜 이미지 디코딩
    
    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])  # 진짜 이미지와 연결
    
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)  # 레이블에 랜덤 노이즈를 추가
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    # 모두 진짜 이미지라고 레이블을 만듦 (사실은 거짓)
    misleading_targets = np.zeros((batch_size, 1))
    
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    
    if start > len(x_train) - batch_size:
        start = 0
    
    # 중간 중간 생성 이미지를 저장
    if stop % 100 == 0:
        gan.save_weights('gan.h5')
        print('step %s에서 판별자 손실 : %s'%(step, d_loss))
        print('step %s에서 적대적 손실 : %s'%(step, a_loss))
        
        img = array_to_img(generated_images[0] * 255, scale = False)
        img.save(os.path.join(save_dir, 'gen_frog' + str(step) + '.png'))
        
        # 비교를 위해 진짜 이미지도 저장
        img = array_to_img(real_images[0] * 255, scale = False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
        
# 이미지 출력
import matplotlib.pyplot as plt

random_latent_vectors = np.random.normal(size = (10, latent_dim))

gen_images = generator.predict(random_latent_vectors)
plt.figure(figsize = (3, 3))

for i in range(gen_images.shape[0]):
    img = array_to_img(gen_images[i] * 255, scale = False)
    plt.figure()
    plt.imshow(img)

plt.show()