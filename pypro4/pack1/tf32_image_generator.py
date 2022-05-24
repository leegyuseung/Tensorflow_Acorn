# 이미지 보강 : 기본 이미지를 회전, 확대축소, 축을 통한 이미지 변환을 하여 이미지 수를 늘림
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_test[:1])

plt.figure(figsize=(10,10))
for c in range(100):
    plt.subplot(10, 10, c+1)
    plt.axis('off')
    plt.imshow(x_train[c].reshape(28,28), cmap='gray')
    
# 이미지 보강 연습
from keras.preprocessing.image import ImageDataGenerator

img_generate = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

augment_size = 100
x_augment = img_generate.flow(np.tile(x_train[0].reshape(28*28), 100).reshape(-1, 28, 28, 1),
                              np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
print(x_augment.shape) # (100, 28, 28, 1)
# print(x_augment)

plt.figure(figsize=(10,10))
for c in range(100):
    plt.subplot(10, 10, c+1)
    plt.axis('off')
    plt.imshow(x_augment[c].reshape(28,28), cmap='gray')
plt.show()

# 합치기
img_generate = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

augment_size = 30000
randidx = np.random.randint(x_train.shape[0], size= augment_size) # 인덱스로 사용할 난수 얻기
x_augment = x_train[randidx].copy()
y_augment = y_train[randidx].copy()

x_augment = img_generate.flow(x_augment,
                              np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))

print(x_train.shape) # (90000, 28, 28, 1)
print(y_train.shape) # (90000, 10)

# 모델
model = tf.keras.models.Sequential([
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1),
                                                  padding='same', activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=2),     # 이미지 크기 반으로 축소
              tf.keras.layers.Dropout(0.3),
              
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=2),
              tf.keras.layers.Dropout(0.3),

              tf.keras.layers.Flatten(),

              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

chkpoint = ModelCheckpoint(filepath = 'tf32.h5', monitor='val_loss', verbose = 0, save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=3) # 일반적으로 patience=10
history = model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose = 2, validation_split=0.2, callbacks=[es, chkpoint])
print(history.history['acc'])
print(history.history['loss'])
