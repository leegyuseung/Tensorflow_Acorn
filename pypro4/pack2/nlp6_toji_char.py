# 기존 소설로 자연어 생성 모델 작성 - 글자 단위 학습

import numpy as np
import random, sys
import tensorflow as tf

fn = "rnn_short_toji.txt"
f = open(fn, 'r', encoding='utf-8')
text = f.read()
# print(text)
f.close()

print('행 수:',len(text)) # 350492
import re
text = re.sub(r"[^가-힣]","",text)
print(set(text))
chars = sorted(list(set(text)))
print(chars)
print('사용 중인 글자 수:', len(chars))

char_indics = dict((c, i) for i, c in enumerate(chars))
print(char_indics)
indics_char = dict((i, c) for i, c in enumerate(chars))
print(indics_char)

# 텍스트를 n개의 글자로 자르고 다음에 오는 글자 인식
maxlen = 30
sentences = []
next_char = []

for i in range(0, len(text) - maxlen, 3):
    # print(text[i: i+maxlen])
    sentences.append(text[i: i+maxlen])
    next_char.append(text[i+maxlen])

print('학습할 구문 수 :', len(sentences)) #116821

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
print(x[:2], x.shape) # (116821, 30, 1423)
print(y[:2], y.shape) # (116821, 1423)

for i, sent in enumerate(sentences):
    # print(sent)
    for t, char in enumerate(sent):
        # print(t, ' ', char)
        x[i, t, char_indics[char]] = 1 # i면 t행 char_indics[char]열에 1 : True
    y[i, char_indics[char]] = 1

print(x[:5])
print(y[:5])

# model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, activation='tanh', input_shape=(maxlen, len(chars))))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))

opti = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

"""
# 참고 : 다항식 분포의 샘플 얻기 -----
print(10 * [0.1])
mul = np.random.multinomial(1, 10*[0.1])
print(mul)
# ---------------------------
"""

# 원본 확률분포의 가중치를 조정하고 새로운 글자의 인덱스를 추출하기 위한 함수
# 확률적 샘플링 처리 함수
def sample_func(preds, variety = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / variety
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # softmax 공식
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for num in range(1, 4): # (1, 100)
    print('\n-------------------')
    print('반복 : ', num)
    
    model.fit(x, y, batch_size = 128, epochs=3, verbose=0) # epochs=충분
    
    # 임의의 텍스트 시작
    start_index = random.randint(0, len(text) - maxlen -1)
    
    for variety in [0.5, 1.0, 1.5]: # 다양한 문장 생성
        print('\n다양성: '+str(variety))
        generated = ''
        sentence = text[start_index:start_index + maxlen]
        generated += sentence
        print('---seed : "'+ sentence + '"')
        sys.stdout.write(generated)

    # 시드를 기반으로 텍스트 생성. 500개의 문자열 생성
    for i in range(500):
        x = np.zeros((1, maxlen, len(chars)))
        
        for t, char in enumerate(sentence):
            x[0, t, char_indics[char]] = 1.0
        
        #다음에 올 문자를 예측(샘플링)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample_func(preds, variety)
        next_char = indics_char[next_index]
        
        # 출력
        generated += next_char
        sentence = sentence[1:]+next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
