# 자동차 관련 데이터로 선형회귀모델 작성
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import seaborn as sns

dataset = pd.read_csv('../testdata/auto-mpg.csv')
print(dataset.head(2))
print(dataset.columns)
del dataset['car name']
print(dataset.columns)
print(dataset.corr())
dataset.drop(['acceleration','model year','origin'], axis ='columns', inplace=True)
# print(dataset.info())

dataset['horsepower'] = dataset['horsepower'].apply(pd.to_numeric, errors ='coerce')    #error ignore
print(dataset.info())
print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.info())

print(dataset.head(2))

# sns.pairplot(dataset[['mpg','cylinders','displacement','horsepower','weight']])
# plt.show()

# train/ test split: smaple() method
train_data = dataset.sample(frac = 0.7, random_state=123)
test_data = dataset.drop(train_data.index)
print(train_data[:2], train_data.shape)     # (274, 5)
print(test_data[:2], test_data.shape)       # (118, 5)

train_stat = train_data.describe()
train_stat.pop('mpg')
train_stat = train_stat.transpose()
print(train_stat)

# label : mpg(연비)
train_label = train_data.pop('mpg')
test_label = test_data.pop('mpg')
print(train_label[:2])
print(test_label[:2])

#feature 에 대해서 표준화 진행
# print (train_data[:2])
def st_func(x):     # (요소값 - 평균) / 표준편차
    return (x -train_stat['mean']) / train_stat['std']

# print(st_func(10))
# print(st_func(train_data[:2]))
st_train_data = st_func(train_data)
st_test_data = st_func(test_data)

print(st_train_data[:3])
print(train_label[:3])


print('------------------------------------------------')
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    network = Sequential([
        Dense(units= 64, activation= tf.nn.relu, input_shape=[4] ),
        Dense(units= 64, activation= tf.nn.relu),
        Dense(units= 1, activation= 'linear'),
    ])
    
    # opti = tf.keras.optimizers.RMSprop(0.001)
    opti = tf.keras.optimizers.Adam(0.001)
    network.compile(optimizer=opti, loss = 'mean_squared_error',
                     metrics=['mean_squared_error','mean_absolute_error']) #mse, mae
    return network

model = build_model()
print(model.summary())

# fit 전에 predict() 가능 : 결과는 기대하지 말자

epochs = 5000 # 조기종료 할 것이니까 겁먹지 말자

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode='auto', patience = 5)
# patience에 준 5회 이내에 loss 값이 감소하지 않으면 시스템이 Early_Stopping 함수 콜백해서 중단

history = model.fit(st_train_data, train_label, batch_size=32, epochs=epochs, validation_split=0.2,
                    verbose=1, callbacks=[early_stop])
df = pd.DataFrame(history.history)
print(df.columns)
print(df.head(2))

# 모델 성능(mae, mse) 시각화
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8,12))
    
    plt.subplot(2,1,1)
    plt.xlabel('epochs')
    plt.ylabel('mae [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'train err')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'val err')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.xlabel('epochs')
    plt.ylabel('mse [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'train err')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'val err')
    plt.legend()
    plt.show()
    
plot_history(history)