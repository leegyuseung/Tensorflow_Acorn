# 단순선형회귀모델 작성방법 세 가지 경험하기
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

# 공부 시간에 따른 성적 데이터
x_data = np.array([1,2,3,4,5], dtype=np.float32)
y_data = np.array([11,32,53,64,70], dtype=np.float32)
print(np.corrcoef(x_data, y_data))

print('모델 작성방법1 : Sequential api : 가장 일반적익 단순한 방법')
model = Sequential()
model.add(Dense(units=2, input_dim = 1, activation ='linear'))
model.add(Dense(units=1, activation ='linear'))
print(model.summary())
opti = tf.keras.optimizers.Adam(learning_rate = 0.1)
model.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metric = model.evaluate(x=x_data, y=y_data)
print('loss metrics:',loss_metric)

from sklearn.metrics import r2_score
print('설명력:',r2_score(y_data, model.predict(x_data)))
print('실제값:', y_data)
print('예측값:', model.predict(x_data).flatten())

new_data = [1.5, 2.3, 5.8]
print('새 예측값:', model.predict(new_data).flatten())

"""# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.plot(x_data, model.predict(x_data), 'b', x_data, y_data, 'ko')
plt.show()

# plt.plot(history.history['loss'], label='손실' )
# plt.xlabel('학습 횟수')
plt.plot(history.history['mse'], label='평균제곱오차' )
plt.xlabel('학습 횟수')
plt.legend()
plt.show()"""

# 방법2 : 유향그래프 가능하다.
print('모델 작성방법2 : Function api : 유연한 구조, 입력 데이터로 여러 층을 공유하거나 다양한 종류의 입출력 사용이 가능')
from keras.layers import Input
from keras.models import Model

# 각 층은 일종의 함수처럼 처리함
inputs = Input(shape = (1,)) # tuple
# outputs = Dense(1, activation = 'linear')(inputs) # 이전층을 입력으로 사용하고 변수에 할당한다.
output1 = Dense(2, activation = 'linear')(inputs) # 1개로 들어와 2개로 나가는 것
outputs = Dense(1, activation = 'linear')(output1)

model2 = Model(inputs, outputs)
# 이하는 방법1과 동일
print(model2.summary())
opti = tf.keras.optimizers.Adam(learning_rate = 0.1)
model2.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metric = model2.evaluate(x=x_data, y=y_data)
print('loss metrics2:',loss_metric)

from sklearn.metrics import r2_score
print('설명력2:',r2_score(y_data, model2.predict(x_data)))
print('실제값2:', y_data)
print('예측값2:', model2.predict(x_data).flatten())

# 방법3 : 자연어 처리할 때 사용한다.
print('모델 작성방법3 : Sub classing : 동적인 구조의 네트워크, 고난이도의 작업에서 활용성이 높음')
x_data = np.array([[1],[2],[3],[4],[5]], dtype=np.float32)
y_data = np.array([11,32,53,64,70], dtype=np.float32)

class MyModel(Model):
    def __init__(self): # 생성자에서 필요한 레이어를 생성해준다.
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation = 'linear')
        self.d2 = Dense(1, activation = 'linear')
        
    def call(self, x): # fit, evaluate, predict 할 때 호출한다.
        inputs = self.d1(x)
        return self.d2(inputs)
    
model3 = MyModel()

# 이하는 방법1과 동일
opti = tf.keras.optimizers.Adam(learning_rate = 0.1)
model3.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metric = model3.evaluate(x=x_data, y=y_data)
print('loss metrics3:',loss_metric)

from sklearn.metrics import r2_score
print('설명력3:',r2_score(y_data, model3.predict(x_data)))
print('실제값3:', y_data)
print('예측값3:', model3.predict(x_data).flatten())        
print(model3.summary())

print('모델 작성방법3-1 : custom layer + sub classing : 동적인 구조의 네트워크, 고난이도의 작업에서 활용성이 높음')
from keras.layers import Layer

# custom layer : 사용자 정의층 사용
class Linear(Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape): # call 호출, 가중치 관련 내용을 적을 수 있다.
        self.w = self.add_weight(shape = (input_shape[-1], self.units), 
                                 initializer='random_normal', trainable = True) #trainable = True 역전파 준다. 
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)  # b값은 w의 출력 값과 맞춰준다.
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b # y = wx + b
    
class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        #self.linear1 = Linear(1)
        self.linear1 = Linear(2)
        self.linear2 = Linear(1)
        
    def call(self, inputs):    
        # return self.linear1(inputs)
        x = self.linear1(inputs)
        return self.linear2(x)
    
model4 = MLP()
# 이하는 방법1과 동일
opti = tf.keras.optimizers.Adam(learning_rate = 0.1)
model4.compile(optimizer = opti, loss='mse', metrics=['mse'])
history = model4.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metric = model4.evaluate(x=x_data, y=y_data)
print('loss metrics4:',loss_metric)

from sklearn.metrics import r2_score
print('설명력4:',r2_score(y_data, model4.predict(x_data)))
print('실제값4:', y_data)
print('예측값4:', model4.predict(x_data).flatten())        
print(model4.summary())

tf.keras.utils.plot_model(model, 'abc.png')