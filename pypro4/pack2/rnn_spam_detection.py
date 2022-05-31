# 스팸 메일을 RNN을 사용
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/spam.csv", encoding='latin1')
print(data.head(2))
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
print(data.head(2))
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
print(data.head(2))
print(set(data.v1))
print(data.info())

# v2 열에 중복 데이터 검사
print(data['v2'].nunique()) # 유니크 갯수를 반환  5572 - 5169 만큼 중복 데이터 있음
data.drop_duplicates(subset=['v2'], inplace=True)
print('전체 행 수 :', len(data))

# 레이블 값 시각화
print(data['v1'].value_counts()) # 0 : 4516, 1 : 653

import matplotlib.pyplot as plt
data['v1'].value_counts().plot(kind='bar')
plt.show()

print(data.groupby('v1').size())

# feature, label로 분리
x_data=data['v2']
y_data=data['v1']

# 토큰화, 인코딩
from keras.preprocessing.text import Tokenizer
tok = Tokenizer()
tok.fit_on_texts(x_data)
sequences = tok.texts_to_sequences(x_data)

print(x_data[:5])
print(sequences[:5]) # [[47, 433, 4013, 780, 705, 662, 64, 

word_to_index = tok.word_index
print(word_to_index) #{'i': 1, 'to': 2, 'you': 3, 'a': 4, 등장 빈도수가 많을수록 숫자는 작아진다.

# 등장빈도로 비율 확인
threshold = 2
total_cnt = len(word_to_index)
print('total_cnt:', total_cnt) # 8920
rare_cnt = 0
total_freq = 0 # 전체 단어 빈도수 총합
rare_freq = 0 # threshold 보다 작은 단어 등장 빈도 수 총합

# 단어와 빈도 수 얻기
for  k, v in tok.word_counts.items():
    # print('k:{}, v:{}'.format(v,k))
    total_freq = total_freq + v

    if v < threshold :
        rare_cnt += 1
        rare_freq += v

print('등장빈도가 %s 이하인 단어 수 : %s'%(threshold - 1, rare_cnt)) # 4908
print('등장빈도 1회인 단어 수 비율 :', (rare_cnt / total_cnt) * 100) # 55.022
print('전체 등장빈도에서 등장빈도 1회인 단어 비율 : ', (rare_freq / total_freq) *100) # 6.082
# 위의 결과로 등장빈도 1회인 단어는 제외해도 문제 없다.

tok = Tokenizer(num_words = total_freq - rare_cnt + 1) # 등장빈도가 1 이하인 단어는 제외하고 토큰화
vocab_size = len(word_to_index) + 1
print('단어 집합 수 : ',vocab_size) # 8921

# pad_sequence
# 데이터 길이를 시각화 확인
x_data = sequences
print('메일의 최대 길이:',  max(len(i) for i in x_data))
print('메일의 평균 길이:', (sum(map(len,x_data))/len(x_data)))

plt.hist([len(i) for i in x_data], bins= 50)
plt.xlabel('length of sample')
plt.ylabel('number of sample')
plt.show()

from keras.utils import pad_sequences
max_len = 189
data = pad_sequences(x_data, maxlen=max_len)
print(data.shape) #(5169, 189)
print(data[:1]) 

# train / test split 8 : 2
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
print('n_of_train:',n_of_train ) # 4135
print('n_of_test:', n_of_test) # 1034

import numpy as np
x_train = data[:n_of_train]
y_train = np.array(y_data[:n_of_train])
x_test = data[n_of_train:]
y_test = np.array(y_data[n_of_train:])
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # (4135, 189) (4135,) (1034, 189) (1034,)
print(x_train[:1])
print(y_train[:1])

# 스팸메일 분류 model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(LSTM(32, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2 )
print('test score(loss):%.4f'%(model.evaluate(x_test, y_test)[0]))
print('test score(acc):%.4f'%(model.evaluate(x_test, y_test)[1]))

# 시각화
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], label ='loss')
plt.plot(epochs, history.history['val_loss'], label ='val_loss', c='r')
plt.xlabel('epoch')
plt.legend()
plt.show()

# pred
pred = model.predict(x_test)
print("예측값:",np.where(pred[:1]>0.5,1,0))
print("실제값:",y_test[:1])