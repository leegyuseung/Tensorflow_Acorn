# LSTM으로 주식 예측
# 삼성전자 : 코드 005930

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import FinanceDataReader as fdr

STOCK_CODE = '005930'
stock_data = fdr.DataReader(STOCK_CODE)
print(stock_data.head(3))
print(stock_data.shape)
print(stock_data.tail(3))
print(stock_data.index)

stock_data['year'] = stock_data.index.year
stock_data['month'] = stock_data.index.month
stock_data['day'] = stock_data.index.day
print(stock_data.head(3))

# 시각화
plt.figure(figsize=(10,6))
sns.lineplot(y=stock_data['Close'], x=stock_data.index)
plt.xlabel('time')
plt.ylabel('price')
plt.show()

time_steps = [['2000','2005'],['2005','2010'],['2010','2015'],['2015','2022']]

fig, axes = plt.subplots(2,2)
fig.set_size_inches(10, 6)
for i in range(4):
    ax = axes[i//2, i%2]
    df = stock_data[(stock_data.index > time_steps[i][0]) & (stock_data.index < time_steps[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]} ~ {time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')

plt.tight_layout()
plt.show()

# 데이터 전처리
# 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_cols = ['Open','High','Low','Close','Volume']
scaled = scaler.fit_transform(stock_data[scale_cols])
# print(scaled)

df = pd.DataFrame(scaled, columns=scale_cols)
print(df.head(3))

print(df.shape) # (6000, 5)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close',1), 
                                                    df['Close'], test_size=0.2, random_state=0, shuffle=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train[:2])
print(y_train[:2])

#tensorflow dataset을 이용해서 sequential dataset을 운영
import tensorflow as tf

def window_dataset_func(series, window_size, batch_size, shuffle_tf):
    series = tf.expand_dims(series, axis= -1) # 차원확대
    # print(series.shape)
    ds = tf.data.Dataset.from_tensor_slices(series) #slice
    ds = ds.window(window_size +1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w:w.batch(window_size+1)) # Dataset 객체를 객체화
    if shuffle_tf:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w : (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1) # 훈련 속도 증진, 학습 성능 향상
    # 대량의 데이터를 메모리에 로딩할 수 없으므로 batch 단위로 읽어 처리하기 위해 준비


WINDOW_SIZE = 20 # 20일 단위로 비교
BATCH_SIZE = 32

# train_data, test_data를 생성
train_data = window_dataset_func(y_train,WINDOW_SIZE,BATCH_SIZE,True)
test_data = window_dataset_func(y_test,WINDOW_SIZE,BATCH_SIZE,False) # test_data는 섞지 않는다.

for data in train_data.take(1):
    print('dataset 구성(batch_size, window_size, feature) : ', data[0].shape)
    print('dataset 구성(batch_size, window_size, feature) : ', data[1].shape)
    
# 모델
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM
from keras.losses import Huber # cost function의 일종으로 mse 보다 이상치에 덜 민감함
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
       # 1차원 feature map 생성
       Conv1D(filters=32, kernel_size=5, padding='causal',
              activation='relu', input_shape=[WINDOW_SIZE, 1]), # padding='casual' 모델이 시간 순서를 지켜야하느 경우의 모델 생성           
       LSTM(16, activation='tanh'),
       Dense(16, activation='relu'),
       Dense(1)
])

loss = Huber()
optimizer = Adam(learning_rate = 0.0005)
model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
print(model.summary())

pred = model.predict(test_data)
print('pred.shape:' , pred.shape)
print('pred:',pred[:10].flatten())
print('real:',np.array(y_test)[:10])

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(np.asarray(y_test)[20:], label='real')
plt.plot(pred, label='predict')
plt.legend()
plt.show()