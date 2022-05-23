# fashionmnist > colab으로 실행
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline > jupyter에서 show대신 사용

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Shirt','Sneaker','Bag','Ankel boot']
print(set(train_labels))

plt.imshow(train_images[0], cmap='gray')
plt.show()

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
    plt.imshow(train_images[i])
plt.show()

# 정규화
print(train_images[0])
train_images = train_images / 255.0
print(train_images[0])
test_images = test_images / 255.0

# model
model = tf.keras.Sequential([
   keras.layers.Flatten(input_shape=(28,28)),
   keras.layers.Dense(units=128, activation=tf.nn.relu),
   keras.layers.Dense(units=128, activation='relu'),
   keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossesntropy', metrics=['accuracy'])

print(model.summary())

# 종기 종료 : Class로 처리
#es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

class myEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): #method override
        if logs.get('loss') < 0.25:
            print('\n 학습 종료')
            self.model.stop_training = True

my_callback = myEarlyStopping()

model.fit(train_images, train_labels, batch_size =128, epochs=500, verbose=2, callbacks=[my_callback])

pred = model.predict(test_images)
print(pred[0])
print('예측값:',pred[0])
print('예측값:',np.argmax(pred[0]))
print('실제값:',test_labels[0])

def plot_image(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i],  img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    
    pred_label = np.argmax(pred_arr)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{}{:2.0f}% ({})".format(class_names[pred_label], np.max(pred_arr)*100, 
                                      class_names[true_label], color=color))
  
def plot_value_arr(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisPlot = plt.bar(range(10), pred_arr)
    plt.ylim([0,1])
    pred_label = np.argmax(pred_arr)
    thisPlot[pred_label].set_color('red') #예측값
    thisPlot[true_label].set_color('blue') #실제값
  
# 예측 결과 출력 정보 자세히 보기
i = 0
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plot_image(i, pred, test_labels, test_images)
#plt.show()
plt.subplot(1,2,2)
plot_value_arr(i, pred, test_labels)
plt.show()