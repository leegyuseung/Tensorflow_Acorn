# zoo dataset : 동물을 7가지로 분류
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

xy = np.loadtxt('../testdata/zoo.csv', delimiter = ',')
print(xy[:2],' ',xy.shape) #(101, 17)

x_data = xy[:, 0:-1] # feature
y_data = xy[:, [-1]] # label
print(x_data[:2])
print(y_data[:2], ' ', set(y_data[:,0])) # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

# train / test 생략
# label 원핫 처리 생략
print()

model = Sequential()
model.add(Dense(32, input_shape=(16,), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
# loss='sparse_categorical_crossentropy' 하면 label의 원핫처리를 내부적으로 진행

print(model.summary())

history = model.fit(x_data, y_data, epochs=100, batch_size = 32, verbose=0)
print('evaluate:',model.evaluate(x_data, y_data))

# loss, acc 시각화 
history_dict = history.history

loss = history_dict['loss']
acc = history_dict['acc']

import matplotlib.pyplot as plt

plt.plot(loss, 'b-', label='loss')
plt.plot(acc, 'k-', label='acc')
plt.xlabel('epochs')
plt.legend()
plt.show()

print('predict')
pred_data = x_data[:1]
print('한 개 예측값:', np.argmax(model.predict(pred_data)))

print()
pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값 들', preds)
print('실제값 들', y_data[:5].flatten())
