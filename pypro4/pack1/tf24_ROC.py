# iris dataset으로 분류 모델(복수) 작성 후 성능 비교

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.keys())

x = iris.data # feature
print(x[:2])
y = iris.target # label
print(y[:2])

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label 원핫
onehot = OneHotEncoder(categories = 'auto')
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y[:2], ' ', y.shape)
print()

# feature는 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train / test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.3, random_state = 12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 4) (45, 4) (105, 3) (45, 3)

N_FEATURES = x_train.shape[1] # 4
N_CLASSES = y_train.shape[1] # 3

# model 복수
def create_custom_model(input_dim, output_dim, out_nodes, n, model_name='model'):
    #print(input_dim, output_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name = model_name)
        for _ in range(n):
            model.add(Dense(units=out_nodes, input_dim=input_dim, activation='relu'))
            
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    return create_model

models = [create_custom_model(N_FEATURES, N_CLASSES, 10, n, 'model_{}'.format(n)) for n in range(1,4)]
print(len(models))

# for cre_model in models:
#     print('-------')
#     cre_model().summary()

history_dict = {}

for cre_model in models:
    model = cre_model()
    print('모델명:', model.name)
    historys = model.fit(x_train, y_train, batch_size=5, epochs=50, verbose=0, validation_split=0.3)
    score = model.evaluate(x_test,y_test,verbose=0)
    print('testloss:',score[0])
    print('testacc:',score[1])
    history_dict[model.name] = [historys, model]
    
print(history_dict)

print()
# 성능 시각화
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))

for model_name in history_dict:
    # print('h_d : ', history_dict[model_name][0].history['acc'])
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation acc')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()    

# ROC curve : 분류기에 성능 평가 기법
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel()) #ravel : 차원축소
    plt.plot(fpr, tpr, label='{}, AUC value:{:.3f}'.format(model_name,auc(fpr,tpr)))
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# 이하 가장 좋은 모델로 작업을 계속 진행..