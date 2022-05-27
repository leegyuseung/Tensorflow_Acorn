# 자연어 처리 알고리즘 모델 : RNN - 순서가 있는 형태의 데이터일 경우 효과적인 알고리즘
# RNN 네트워크 구성

from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense

model = Sequential()
# model.add(SimpleRNN(units=3, input_shape=(2,10)))
# model.add(LSTM(units=3, input_shape=(2,10)))
model.add(GRU(units=3, input_shape=(2,10)))
print(model.summary())

print()
model = Sequential()
# model.add(SimpleRNN(units=3, batch_input_shape = (8, 2, 10)))
model.add(LSTM(units=3, batch_input_shape = (8, 2, 10)))
print(model.summary())

print()
model = Sequential()
# model.add(SimpleRNN(units=3, batch_input_shape = (8, 2, 10), return_sequences=True))
model.add(LSTM(units=3, batch_input_shape = (8, 2, 10), return_sequences=True))
print(model.summary())

print('\n4개의 숫자가 주어지면 그 다음 숫자 예측하기 모델 작성---')

x=[]
y=[]
for i in range(6):
    lst = list(range(i,i+4))
    #print(lst)
    x.append(list(map(lambda c:[c/10],lst)))
    y.append((i+4)/10)

import numpy as np
x = np.array(x)
y = np.array(y)
print(x)
print(y)

model = Sequential([
        # SimpleRNN(units=10, activation='tanh', input_shape=[4,1]),
          LSTM(units=10, activation='tanh', input_shape=[4,1]),

        Dense(1)
    ])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

model.fit(x, y, epochs=100, verbose = 2)
print('예측값:', model.predict(x).flatten())
print('실제값:', y)

# 새로운 값으로 결과 예측
print('새예측값:', model.predict(np.array([[[0.6],[0.7],[0.8],[0.9]]])))