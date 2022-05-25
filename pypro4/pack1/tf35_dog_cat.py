# CNN으로 고차원 이미지 분류 : 개, 고양이 이진 분류
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# data download
data_url = "http://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=data_url, extract=True)
path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_dir = os.path.join(path, 'train')
validation_dir = os.path.join(path, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 이미지 확인
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
print(os.listdir(train_cats_dir)[:5])
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total train cat images :', num_cats_tr)
print('total train dog images :', num_dogs_tr)
print('total validation cat images :', num_cats_val)
print('total validation dog images :', num_dogs_val)

print('전체 학습용 데이터 수 :', total_train)
print('전체 검증용 데이터 수 :', total_val)

# 이미지 불러오기
train_image_gen = ImageDataGenerator(rescale = 1./255) # 하드웨어에 저장
validation_image_gen = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_gen.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,  # 이미지를 128장씩 램에 불러 읽는
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode= 'binary') # 크기재조정 라벨링
#class_mode = 'binary'는 labeling을 해준다.
val_data_gen = validation_image_gen.flow_from_directory(batch_size=batch_size, directory=validation_dir, # validation은 굳이 shuffle 안해도 됌
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode= 'binary')

# 데이터 확인 : next() 사용
sample_train_images, _ = next(train_data_gen) #next > 이터러블데이터

def plotImage(img_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

plotImage(sample_train_images[:5])

# !find / -name 'cats_and_dogs*' 
# !ls /root/.keras/datasets/cats_and_dogs_filtered/train/cats/ -la

# model
model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(IMG_HEIGHT,  IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),    
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer = 'adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print(model.summary())

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size, #에폭당 작업 # 기본값은 len(generator)
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps=total_train // batch_size
)

model.save('tf35.hdf5')

# 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train acc')
# plt.plot(epochs_range, val_acc, label='Train val_acc')
plt.legend(loc='best')
plt.title('Train, Validation acc')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train loss')
# plt.plot(epochs_range, val_loss, label='Train val_loss')
plt.legend(loc='best')
plt.title('Train, Validation loss')

plt.show()

# 학습 후 과적합 문제 발생시 : 학습 데이터 수 증가 시키자 - 이미지 보강
image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary') # (128, 150, 150, 3)
augment_image = [train_data_gen[0][0][1] for i in range(5)]
plotImage(augment_image)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size, directory=validation_dir, 
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary') # (128, 150, 150, 3)

# remodel
model_new = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(IMG_HEIGHT,  IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=2),
    Dropout(rate=0.2),
    Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),    
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

model_new.compile(optimizer = 'adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print(model_new.summary())

history = model_new.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size, #에폭당 작업 # 기본값은 len(generator)
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps=total_train // batch_size
)

model_new.save('tf35.hdf5')

# 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train acc')
# plt.plot(epochs_range, val_acc, label='Train val_acc')
plt.legend(loc='best')
plt.title('Train, Validation acc')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train loss')
# plt.plot(epochs_range, val_loss, label='Train val_loss')
plt.legend(loc='best')
plt.title('Train, Validation loss')

plt.show()

# 새로운 이미지로 분류 < 1번모델 사용 < 코랩에서만 사용
# from google.colab import files
# import numpy as np
# from keras.preprocessing import image
#
# uploaded = files.upload()

# for fn in uploaded.keys():
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(150,150))
#
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#
#     pred = model.predict(images, batch_size=10)
#     print(pred)
#     print(pred[0])
#
#     if pred[0] > 0:
#       print(fn + '는 댕댕이')
#     else:
#       print(fn + '는 냥이')