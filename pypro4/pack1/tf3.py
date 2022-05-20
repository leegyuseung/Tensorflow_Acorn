# constant : 텐서를 직접 기억
# Variable : 텐서가 저장된 주소를 참조

import tensorflow as tf
import numpy as np

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
imsi = tf.add(node1, node2)
print(imsi)

print()
node3 = tf.Variable(3, dtype=np.float32)
node4 = tf.Variable(4.0)
print(node3)
print(node4)
imsi2 = tf.add(node3, node4)
print(imsi2)
node4.assign_add(node3)
print(node4)

print()
a = tf.constant(5)
b = tf.Variable(10)
c = tf.multiply(a, b)
print(c, c.numpy())
result = tf.cond(a>b, lambda:tf.add(10,c), lambda:tf.square(a))
print('result : ', result.numpy())

print('-------------')
v = tf.Variable(1)

@tf.function
def find_next_odd():
    v.assign(v+1)
    if tf.equal(v%2, 0):
        v.assign(v+10)

find_next_odd()
print(v.numpy())
print(type(find_next_odd))

print('~~~' * 10)
# imsi = tf.constant(0)
def func1(): # 1 ~ 3 까지 증가
    imsi = tf.constant(0) # imsi = 0
    su = 1
    for _ in range(3):
        # imsi=tf.add(imsi, su)
        # imsi = imsi + su
        imsi+=su
    return imsi
kbs = func1()
print(kbs.numpy(),' ',np.array(kbs))

print()
imsi = tf.constant(0)
@tf.function
def func2(): # 1 ~ 3 까지 증가
    # imsi = tf.constant(0) # imsi = 0
    global imsi
    su = 1
    for _ in range(3):
        # imsi=tf.add(imsi, su)
        # imsi = imsi + su
        imsi+=su
    return imsi

mbc = func2()
print(mbc.numpy(),' ',np.array(mbc))

print('^^^'* 10)
imsi = tf.Variable(0)
@tf.function
def func3():
    #imsi = tf.Variable(0) #autograph 기능이있으면 변수는 밖에다가 해야한다. 
    su = 1
    for _ in range(3):
        imsi.assign_add(su) # 변수가 밖에있을 때 누적방식
        # imsi = imsi + su # 변수가 밖에있으면 누적이 이런방식으로 안된다.
        # imsi += su
    return imsi

sbs = func3()
print(sbs.numpy(), ' ', np.array(sbs))

print('------99단-----')

#@tf.function #연산만 담당, 출력X
def gugu1(dan):
    su = tf.constant(0) # su = 0
    for _ in range(9):
        su = tf.add(su, 1)
        #print(su)
        #print(su.numpy())
        print('{}*{}={:2}'.format(dan, su, dan * su))
gugu1(3)

print()
# @tf.function
def gugu2(dan):
    for i in range(1, 10):
        result=tf.multiply(dan, i) #tf.multiply() : 요소곱, tf.matmul() : 행렬곱
        print('{}*{}={:2}'.format(dan, i, result))
gugu2(4)