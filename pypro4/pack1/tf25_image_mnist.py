# 완전연결층(Dense)으로 이미지 분류
# MNIST dataset : 흑백 손글씨 이미지 7만장과 라벨 7만개
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])
print(x_test[0])

# import sys
# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s'%j)
#     sys.stdout.write('\n')

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

x_train = x_train.reshape(60000,784).astype('float32') # 28 * 28 --> 784 구조 변경
x_test = x_test.reshape(10000,784).astype('float32')
print(x_train[0])
x_train /= 255 # 정규화 : 필수는 아니나 해주면 성능향상
x_test /= 255
print(x_train[0])

# label은 원핫인코딩
print(set(y_train)) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
print(y_train[0])
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

print(y_train[0])

# train_data의 일부를 validation data로 사용하기
x_val = x_train[50000:60000] # 일만개는 validation data
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000]
print(x_train.shape, ' ', x_val.shape)
"""
# model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation = 'relu'))
# model.add(Flatten(input_shape=(28,28))) # reshape 안했을 때 
model.add(Dropout(0.2))
model.add(Dense(units=128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, batch_size = 128, validation_data=(x_val,y_val), verbose = 2)

print('loss:',history.history['loss'])
print('val_loss:',history.history['val_loss'])
print('accuracy:',history.history['accuracy'])
print('val_accuracy:',history.history['val_accuracy'])

# 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

# 모델 평가
score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])

# 모델 저장
model.save('tf25.hdf5')
del model
"""

mymodel = tf.keras.models.load_model('tf25.hdf5')

"""
plt.imshow(x_test[:1].reshape(28,28),  cmap='Greys')
plt.show()
print(x_test[:1], x_test[:1].shape)

# 예측
pred = mymodel.predict(x_test[:1])
print('pred:',pred)
print('예측값:',np.argmax(pred,1))
print('실제값:',np.argmax(y_test[:1]))
"""

# 내가 그린 손글씨 이미지 분류 결과 보기
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('num.png')
img = np.array(im.resize((28,28), Image.ANTIALIAS).convert('L')) # grey scale
print(img.shape)

plt.imshow(img, cmap='Greys')
plt.show()

data = img.reshape([1,784])
data = data / 255.0
print(data)

pred = mymodel.predict(data)
print('이미지 예측값:',np.argmax(pred,1))
