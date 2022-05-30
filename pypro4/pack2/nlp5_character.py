# LSTM : 영문 학습 후 문장 생성 - 단어 단위가 아니라 글자 단위의 학습 모델 생성
filename = 'eng_text.txt'
et = open(filename, encoding='utf-8').read().lower()
print(et)

# 중복 제거 후 글자 인덱싱
chars = sorted(list(set(et)))
print(chars)
char_to_int = dict((c, i )for i, c in enumerate(chars))
print(char_to_int)

n_chars = len(et)
n_vocab = len(chars)
print('전체 글자 수 : ', n_chars)
print('전체 어휘 크기 : ', n_vocab)

# 위에서 만든 dict 데이터로 input, output data를 생성
seq_length = 5
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = et[i : i+seq_length]
    seq_out = et[i+seq_length]
    # print(seq_in, '~', seq_out)
    # hello tom. are you ok ==> hello tom. ~ ello tom. ~ a
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])

print(dataX)
print(dataY)

dataX_patterns = len(dataX)
print('dataX의 행렬 유형 수 :', dataX_patterns)

import numpy as np
from keras.utils.np_utils import to_categorical

# dataX의 구조 변경
feature = np.reshape(dataX, (dataX_patterns, seq_length, 1))
# print(feature, feature.shape)

feature = feature / float(n_vocab) # 정규화
print(feature[:1])

label = to_categorical(dataY)
print(label[:1])

# model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
import matplotlib.pyplot as plt

model = Sequential()
model.add(LSTM(units=256, input_shape=(feature.shape[1], feature.shape[2]),
               activation='tanh', return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=256, activation='tanh'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=label.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(feature, label, batch_size=32, epochs=50, verbose=2)

# 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], label='train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], label='train acc', c='b')
acc_ax.set_ylabel('acc')
acc_ax.legend(loc='lower left')

plt.show()

# 문장작성
int_to_char = dict((i,c) for i, c in enumerate(chars))
print('int_to_char:', int_to_char)

start = np.random.randint(0, len(dataX)- 1)
pattern = dataX[start]
print('pattern:', pattern)

print('seed:')
print("\"",''.join([int_to_char[value] for value in pattern]),"\"")

print()
for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    pred = model.predict(x, verbose = 0)
    # print(pred)
    index = np.argmax(pred)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # print(result)
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print('완료')