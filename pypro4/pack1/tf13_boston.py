# 보스톤 집값 데이터로 분류 모델
import numpy as np
from keras import models, layers 
from keras.datasets import boston_housing
aa = boston_housing.load_data()
# print(aa, type(aa))
(x_train, y_train), (x_test, y_test) = boston_housing.load_data() # 8:2로 분리되어 저장됨
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) #(404, 13) (404,) (102, 13) (102,)

print(x_train[:2])
# print(x_test[:2])
print(y_train[:2])

# # feautre에 대한 스케일링 : 표준화, 정규화
# from sklearn.preprocessing import StandardScaler
# x_train = StandardScaler().fit_transform(x_train)
# print(x_train[:1])
# [[-0.27224633 -0.48361547 -0.43576161 -0.25683275 -0.1652266  -0.1764426
#    0.81306188  0.1166983  -0.62624905 -0.59517003  1.14850044  0.44807713
#    0.8252202 ]]

# 직접 수식 사용 :(요소값 - 평균) / 표준편차
mean = x_train.mean(axis = 0)
x_train -= mean
std = x_train.std(axis = 0)
x_train /= std
print(x_train[:1])

x_test -= mean
x_test /= mean

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='linear', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(32, activation='linear'))
    model.add(layers.Dense(1, activation='linear')) 
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

model = build_model()
print(model.summary())

# 방법1 : train / test split
# history = model.fit(x_train, y_train, epochs=50, batch_size = 10, verbose=2)

# 방법2 : train-validation / test split
history = model.fit(x_train, y_train, epochs=50, batch_size = 10, validation_split=0.2, verbose=2)

mse_history = history.history['mse']
print('mse_history:',mse_history)
val_history = history.history['val_mse']
print('val_history:',val_history)

# import matplotlib.pyplot as plt
# plt.plot(mse_history, 'r', label='mse or loss')
# plt.plot(val_history, 'b', label='validation or val_loss')
# plt.xlabel('epochs')
# plt.ylabel('mse')
# plt.legend()
# plt.show()

# predict
print('예측값:', np.squeeze(model.predict(x_test[:5])))
print('실제값:', y_test[:5])

from sklearn.metrics import r2_score
print('설명력 :', r2_score(y_test, model.predict(x_test)))

print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# concatenate : numpy 배열 결합
a = np.array([[1,2],[3,4]]) # 2 by 2
b = np.array([[5,6],[7,8],[9,10]]) # 3 by 2
print(np.concatenate((a,b), axis=0)) # 열 기준
print(np.concatenate((a,b.T), axis=1)) # 행 기준
print(np.concatenate((a,b), axis=None)) # 첫 번째 배열부터 가로방향 -> 차원 축소
print()

# 모델 학습 및 검정 : k-fold 적용 - 비교적 가용 데이터가 적은 경우에 효과적
k=4
val_samples = len(x_train) // k
all_mse_history = []

for i in range(k):
    # print('i:',i)
    # train data의 일부를 validation data로 사용하기 위해 데이터 추출 : 전체 데이터 수 -404
    # print(i * val_samples, ':', (i+1)*val_samples)#0 : 101 101 : 202 202 : 303 303 : 404
    val_x = x_train[i*val_samples:(i+1) *val_samples] # validation data
    val_y = y_train[i*val_samples:(i+1) *val_samples]
    # print(val_x.shape, ' ', val_y.shape) #(101, 13)   (101,)
    # print(val_x[:1])
    
    # validation data를 제외한 나머지는 train
    train_x = np.concatenate([x_train[:i*val_samples], x_train[(i+1)*val_samples:]], axis=0)
    train_y = np.concatenate([y_train[:i*val_samples], y_train[(i+1)*val_samples:]], axis=0)
    #print(train_x.shape, ' ', train_y.shape) #(303, 13)   (303,)
    
    model2 = build_model()
    history2 = model2.fit(train_x, train_y, epochs=50, batch_size = 10,
                         validation_data=(val_x,val_y), verbose=2)
    mse_history2 = history2.history['mse']
    print('mse_history2:',mse_history2)
    val_history2 = history2.history['val_mse']
    print('val_history2:',val_history2)
    all_mse_history.append(mse_history2)

import matplotlib.pyplot as plt
plt.plot(mse_history2, 'r', label='mse or loss')
plt.plot(val_history2, 'b', label='validation or val_loss')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.show()

print('kfold 전체 평균:', np.mean(all_mse_history))
from sklearn.metrics import r2_score
print('설명력 :', r2_score(y_test, model2.predict(x_test)))
