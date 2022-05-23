''' Created on 2022. 5. 23. 1교시 ~ '''
# 지난주 수업 정리하는 중~~
# bmi dataset으로 분류
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

bmi = pd.read_csv('bmi.csv')
print(bmi.head(2), bmi.shape)   # (50000, 3)

# 정규화
bmi['height'] /= 200
bmi[' weight'] /= 100
print(bmi.head(2))

x = bmi[['height', ' weight']].values
print(x[:2])

# label은 One-Hot-Encoding 처리
bclass = {'thin':[1,0,0], 'normal':[0,1,0], 'fat':[0,0,1]}
y = np.empty((50000, 3))    # tuple로 생성
print(y[:2])

# y애 bclass 넣기
for i, v in enumerate(bmi[' label']):    # i가 순서(index), v는 값(value)
    y[i] = bclass[v]
print(y[:2])

# train / test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (35000, 2) (15000, 2) (35000, 3) (15000, 3)

print()
# model
model = Sequential()
model.add(Dense(128, input_shape = (2,), activation='relu'))
# input_shape은 height, weight이 두 개가 입력되는 것.
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))   # 분류를 끝낼 때 활성함수는 결과값이 3개 이상일 때는 softmax 쓴다.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

# Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='auto', baseline=0.05, patience=5)  # cost가 baseline값에 도달하면 종료.

# 학습
model.fit(x_train, y_train, batch_size = 64, epochs=10000, validation_split=0.2, 
          verbose=2, callbacks=[es])
# batch_size 크면 속도 향상 | train의 20%는 학습 도중에 검증 데이터로 사용한다

# evaluate
m_score = model.evaluate(x_test, y_test)
print('loss : ', m_score[0])
print('accuracy : ', m_score[1])

# predict
print('예측값 : ', np.argmax(model.predict(x_test[:1]), axis=1))
print('실제값 : ', np.argmax(y_test[:1]))

# 새로운 예측값
print('예측값 : ', np.argmax(model.predict(np.array([[187/200, 55/100]])), axis=1))
print('예측값 : ', np.argmax(model.predict(np.array([[157/200, 75/100]])), axis=1))
