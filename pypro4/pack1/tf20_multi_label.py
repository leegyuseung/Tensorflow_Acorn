# 다항분류 : 출력값이 softmax 함수를 통해 확률 값으로 여러 개가 출력. 이 중에서 확률값이 가장 큰 인덱스를 분류의 결과로 얻음
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical #one-hot encoding을 지원
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1)
# tf.random.set_seed(1)

xdata = np.random.random((1000,12))
ydata = np.random.randint(5, size=(1000,1)) # 범주 5개, 시험과목 0:국어 ~ 4:체육라고 가정
print(xdata[:2])
print(ydata[:2]) # [[2][0]] 학습시 label은 원핫 처리 후 학습에 참여
ydata = to_categorical(ydata, num_classes = 5)
print(ydata[:2])
print(np.argmax(i) for i in ydata[:2])

# 모델
model = Sequential()
model.add(Dense(units = 32, input_shape = (12,), activation='relu'))
model.add(Dense(units = 16, activation='relu'))
model.add(Dense(units = 5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history= model.fit(xdata, ydata, epochs=2000, batch_size=32, verbose=2, shuffle=True)

model_eval = model.evaluate(xdata, ydata)
print('model_eval :', model_eval)

# 시각화
plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

# 예측
print('예측값:',model.predict(xdata[:5]))
print('예측값:',[np.argmax(i) for i in model.predict(xdata[:5])])
print('예측값:',ydata[:5])
print('예측값:',[np.argmax(i) for i in ydata[:5]])

# 새로운 값으로 예측
x_new = np.random.random([1,12])
print(x_new)
new_pred = model.predict(x_new)
print('분류 결과:', new_pred)
print('분류 결과:', np.sum(new_pred))
print('분류 결과:', np.argmax(new_pred))

CLASSES = np.array(['국어','영어','수학','과학','체육'])
print('분류결과', CLASSES[np.argmax(new_pred)])