# GAN : 생성적 적대 신경망
# 어떤 분포 혹은 분산 자체를 만들어 내는 알고리즘 모델
# 사진복원, 허구 이미지 생성, 웹툰 이미지, 음성 합성, 태아의 초음파 사진으로 아기의 얼굴 예측 ...
# input data(노이즈가 엄청 심함) --> 생성자 : 비지도학습 --> fake data --> 판별자 판별 : 지도학습
#                                                        --> real data
# MNIST dataset으로 작업
# https://thebook.io/080228/part05/ch19/ 링크 참조(해설)
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, LeakyReLU, Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
if not os.path.exists("./gan_images"):
    os.makedirs("./gan_images")

np.random.seed(3)
tf.random.set_seed(3)

# 생성자 모델
generator = Sequential()   # 모델 이름을 generator로 정하고 Sequential() 함수를 호출
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization()) # 데이터의 배치를 정규분포로 만듦, 안정적 학습을 유도
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D()) # 이미지의 크기를 2배 확장

generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))

print(generator.summary())

# 판별자
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False # 판별자는 학습하지 않는다.
# 판별자는 진짜/가짜 만 판별할 뿐 자기자신은 학습하지 않는다.
# 판별자가 얻은 가중치는 판별자 자신이 학습하는데 쓰이는것이 아니라 생성자로 넘겨 주어 생성자가 업데이트된 이미지를 만들도록 함

# 생성자와 판별자 모델을 연결시키는 gan 모델 만들기 
ginput = Input(shape=(100,))
 
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

# 신경망을 실행시키는 함수 만들기
def gan_train(epoch, batch_size, saving_interval):

# MNIST 데이터 불러오기
    # 앞서 불러온 MNIST를 다시 이용, 테스트 과정은 필요없고 이미지만 사용할 것이기 때문에 X_train만 호출
    (X_train, _), (_, _) = mnist.load_data()  
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    # 127.5를 빼준 뒤 127.5로 나눠서 -1~1사이의 값으로 바꿈
    X_train = (X_train - 127.5) / 127.5  
    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for i in range(epoch):
        # 실제 데이터를 판별자에 입력
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)  # 배치 크기 만큼 판별을 시작
        
        # 가상 이미지를 판별자에 입력
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)  # gen_imgs에 모두 가짜(0) 라는 레이블이 붙는다.
        
        # 판별자와 생성자의 오차 계산
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)  # 생성자의 레이블은 무조건 참이라 해놓고 판별자에 넘김
        
        print('생성자와 소멸자의 오차 epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)
        
        # 중간 과정을 이미지로 저장하는 부분. 정해진 인터벌만큼 학습되면 그때 만든 이미지를 gan_images 폴더에 저장하라는 뜻. 이 코드는 본 장의 주된 목표와는 관계가 없어서 소스 코드만 소개한다
        
        if i % saving_interval == 0:
        # r, c = 5, 5
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
     
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
      
            fig, axs = plt.subplots(5, 5)  
            count = 0
            for j in range(5):
                for k in range(5):
                    axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[j, k].axis('off')
                    count += 1
                    fig.savefig("gan_images/gan_mnist_%d.png" % i)
  
# 4,000번 반복되고(+1을 하는 것에 주의), 배치 크기는 32, 500번마다 결과가 저장됨
gan_train(4001, 32, 500)