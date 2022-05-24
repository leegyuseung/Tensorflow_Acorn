# 칼라 이미지로 연습
# cifr10은 50000:train, 10000:test 이미지와 label
# label은 10가지 : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# 32 * 32 color image

# 연습2 : CNN O
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D, ReLU, LeakyReLU, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
from keras.datasets import cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
print(x_train[0])
print(y_train[0])

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.imshow(x_train[0], interpolation='bicubic')
plt.subplot(132)
plt.imshow(x_train[1], interpolation='none')
plt.subplot(133)
plt.imshow(x_train[2], interpolation='nearest')
plt.show()

x_train = x_train.astype('float32') / 255.0 #정규화
x_test = x_test.astype('float32') / 255.0 
NUM_CLASSES = 10
y_train = to_categorical(y_train, NUM_CLASSES) #완화처리
y_test = to_categorical(y_test, NUM_CLASSES)
# print(x_train[[0]])
print(x_train[54, 12, 23, 1]) # R : 0 G : 1 B : 2
# 인덱스 54번째의 12행 23열 위치의 초록색 값 : 0.35686275 
print(y_train[[0]])

# model : CNN O - function api
input_layer = Input((32,32,3))
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(input_layer)
x = LeakyReLU()(x)
x = BatchNormalization()(x) # 과적합 방지 목적 - 배치 정규화 : 가중치 초기값 선택의 의존성이 적어짐 Dropout 대체, 같이쓰지않는다.
x = MaxPool2D(pool_size=2)(x) # 단순히 크기만 줄이는 역할

x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x) 
x = MaxPool2D(pool_size=2)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(input_layer, output_layer)

print(model.summary())

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, verbose=2)
print('test loss: %.4f'%(model.evaluate(x_test, y_test, verbose = 0, batch_size=128)[1]))
print('test acc: %.4f'%(model.evaluate(x_test, y_test, verbose = 0, batch_size=128)[0]))

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
pred = model.predict(x_test[:10])
pred_single = CLASSES[np.argmax(pred, axis= -1)]
actual_single = CLASSES[np.argmax(y_test[:10], axis= -1)]
print('예측값:', pred_single)
print('실제값:', actual_single)
print('분류 실패 수 :',  (pred_single != actual_single).sum())

# 시각화
fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, idx in enumerate(range(len(x_test[:10]))):
    img = x_test[idx]
    ax = fig.add_subplot(1, len(x_test[:10]),i+1)
    ax.axis('off')
    
    ax.text(0.5, -0.35, 'pred='+str(pred_single[idx]),fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act='+str(actual_single[idx]),fontsize=10, ha='center', transform=ax.transAxes)
    
    ax.imshow(img)

plt.show()