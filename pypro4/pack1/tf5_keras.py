# Keras : 텐서플로 기반의 DeepLearning 라이브러리(모듈)
# 일관성 있는 API를 지원
# ML(인공신경망) 모델을 매우 쉽게 작성할 수 있다.

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

# OR 게이트 논리 모델을 생성 후 처리
# 1. 데이터 셋 생성
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])
print(x) # feature
print(y) # label(class)

# 2. 모델 구성
# model = Sequential([
#     Dense(input_dim = 2, units = 1),
#     Activation('sigmoid')
# ])

model = Sequential()
model.add(Dense(units=1,input_dim = 2))
model.add(Activation('sigmoid'))

# 3. 모델 학습과정 설정 (컴파일)
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.9),
#                loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
#                loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
               loss='binary_crossentropy', metrics=['accuracy'])

# optimizer => 경사하강법, RMSprop => momentum 자동, Adam => 가장많이사용

# 4. 모델 학습시키기
model.fit(x, y, epochs=10, batch_size=1, verbose=2)

# 5. 모델 평가
loss_metrics = model.evaluate(x, y)
print('loss_metrics:', loss_metrics)

# 모델을 사용해 예측하기
pred = model.predict(x)
# print('예측결과:', pred)
# print('예측결과:', (pred > 0.5).astype('int32'))
print('예측결과:', (model.predict(x) > 0.5).astype('int32'))

# 모델 성능이 우수하다고 판단 되면 모델을 저장
model.save('test.hdf5')

# del model

# 저장된 모델 읽기
from keras.models import load_model
model2 = load_model('test.hdf5')
print('예측결과:', (model2.predict(x)>0.5).astype('int32'))