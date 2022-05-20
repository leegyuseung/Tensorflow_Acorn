# 지역별 인구증가율과 고령인구비율(통계청 시각화 자료에서 발췌) 데이터로 선형회귀분석 및 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
print(len(population_inc))
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

# 지역별 인구증가율과 고령인구비율 : 이상(극단)치 제거 - 세종시 데이터
population_inc = population_inc[:5] + population_inc[6:]  # 5번째는 제외
population_old = population_old[:5] + population_old[6:]
print(len(population_inc))

# plt.plot(population_inc,population_old,'bo')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

print("방법1: 최소제곱법으로 회귀선 구하기 y = ax + b")
# population_inc, population_old 평균
x_bar = sum(population_inc) / len(population_inc)
y_bar = sum(population_old) / len(population_old)
a = sum([(y - y_bar)*(x - x_bar) for y, x in list(zip(population_old, population_inc))])
a /= sum([(x - x_bar)**2 for x in population_inc])
b = y_bar - a * x_bar
print('a:',a ,',b:',b) # 회귀식이 완성

import numpy as np 
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a*line_x + b

# plt.plot(population_inc,population_old,'bo')
# plt.plot(line_x,line_y, 'r-')
# plt.xlabel('지역별 인구증가율 (%)')
# plt.ylabel('고령인구비율 (%)')
# plt.show()

print("\n방법2: 최소제곱법 X, tensorflow로 회귀선 구하기")
import tensorflow as tf
import random

a = tf.Variable(random.random()) # 난수로 초기화
b = tf.Variable(random.random()) 

# 잔차의 제곱의 평균을 반환
def compute_lossfunction():
    y_pred = a * population_inc + b
    loss = tf.reduce_mean((population_old - y_pred) **2)
    return loss 

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.07)
for i in range(1000):
    optimizer.minimize(compute_lossfunction, var_list = [a, b]) #잔차제곱의평균을최소화한다.
    if i % 100 == 99:
        print(i, 'a:', a.numpy(), ',b:', b.numpy(), ',loss:', compute_lossfunction().numpy())

line_x = np.arange(min(population_inc),max(population_inc), 0.01)
line_y = a*line_x +b
plt.plot(population_inc,population_old,'bo')
plt.plot(line_x,line_y, 'r-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

print('------------비선형 처리 ----------------')
# 다항회귀 (Polynomial Regression) : 비선형 회귀는 다항식을 사용
# 텐서플로로 2차함수로 구성된 회귀선 얻기

a = tf.Variable(random.random()) # 난수로 초기화
b = tf.Variable(random.random()) 
c = tf.Variable(random.random())
 
# y = ax + b 가 아니라 ax² + bx + c 
def compute_lossfunction2():
    y_pred = a * population_inc * population_inc + b * population_inc + c
    loss = tf.reduce_mean((population_old - y_pred) **2)
    return loss 

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.07)
for i in range(1000):
    optimizer.minimize(compute_lossfunction2, var_list = [a, b, c]) #잔차제곱의평균을최소화한다.
    if i % 100 == 99:
        print(i, 'a:', a.numpy(), ',b:', b.numpy(), ',loss:', compute_lossfunction2().numpy())

line_x = np.arange(min(population_inc),max(population_inc), 0.01)
line_y = a * (line_x * line_x) + b * line_x + c

plt.plot(population_inc,population_old,'bo')
plt.plot(line_x,line_y, 'g-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()

print('------------비선형 처리2 ----------------')
# Keras로 네트워크 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 6, activation='relu', input_shape = (1,)),
    tf.keras.layers.Dense(units = 1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1), loss='mse')
model.summary()

model.fit(population_inc,population_old, epochs=50)
print(model.predict(population_inc).flatten())

line_x = np.arange(min(population_inc),max(population_inc), 0.01)
line_y = model.predict(line_x)

plt.plot(population_inc,population_old,'bo')
plt.plot(line_x,line_y, 'k-')
plt.xlabel('지역별 인구증가율 (%)')
plt.ylabel('고령인구비율 (%)')
plt.show()