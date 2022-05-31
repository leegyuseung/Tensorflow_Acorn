# 기존 소설로 자연어 생성 모델 작성 - 단어 단위 학습

import numpy as np
import random, sys
import tensorflow as tf

path_to_file = tf.keras.utils.get_file('toji.txt', 'https://raw.githubusercontent.com/pykwon/etc/master/rnn_short_toji.txt')
train_text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(train_text)
print('전체 글자 수: ', len(train_text)) # 351744
print(train_text[:100])

# 훈련 데이터 정제
import re
def clean_str(string): 
    string = re.sub(r"[^가-힣A-Zz0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    string = re.sub(r"\'", " ", string)       
    string = re.sub(r"\\", " ", string)      
    return string

train_text = train_text.split('\n')
train_text = [clean_str(sentence) for sentence in train_text]
print(train_text)

train_text_x = []
for sentence in train_text:
    train_text_x.extend(sentence.split(' '))
    train_text_x.append('\n')

train_text_x = [word for word in train_text_x if word != ''] #공백이없으면 단어처리
print(train_text_x[:20])

# 단어 토큰화
vocab = sorted(set(train_text_x))
vocab.append('UNK') # 텍스트 안에 존재하지 않는 토큰은 'UNK' 처리
print('사전 크기:{}'.format(len(vocab)))

word2idx = {u:i for i, u in enumerate(vocab)}
print(word2idx)
idx2word = np.array(vocab)
print(idx2word)

text_as_int = np.array([word2idx[c] for c in train_text_x])
print(text_as_int)

print(train_text_x[:20])
print(text_as_int[:20])

# 기본 dataset 만들기
seq_length = 25 # 25개의 단어가 주어질 경우에 다으 단어를 예측하도록 ...
example_per_epoch = len(text_as_int) // seq_length
print(example_per_epoch) # 3847

sentence_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# seq_length + 1 : 25개 단어와 정답단어 1개를 합쳐 반환하기 위함
sentence_dataset = sentence_dataset.batch(seq_length + 1, drop_remainder=True) # 단일 원소를 n개 만큼 쌓기

for item in sentence_dataset.take(1):
    print(item.numpy()) # [ 2762 11139 30139 27877  9051 30541 26908 23209  3645   861 22746
    print(idx2word[item.numpy()]) # ['귀녀의' '모습을' '한번' '쳐다보고' '떠나려' '했다' '집안을' '이리저리' '기웃거리던'

# train dataset 작성 : 26개의 단어가 각각 입력과 정답으로 묶여 ([25단어], 1단어) 형태의 데이터 반환 함수
def split_input_target(chunk):
    return [chunk[:-1], chunk[-1]]

train_dataset = sentence_dataset.map(split_input_target)
for x, y in train_dataset.take(1):
    print(idx2word[x.numpy()])
    print(x.numpy()) # 25개의 단어에 대한 숫자(단어표현)
    print(idx2word[y.numpy()])
    print(y.numpy()) # 다음 단어 예측용 숫자

# dataset shuffle, batch 설정
BATCH_SIZE = 64
steps_per_epoch = example_per_epoch // BATCH_SIZE
BUFFER_SIZE =  5000
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# model
total_words = len(vocab)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length = seq_length),
    tf.keras.layers.LSTM(units=100, return_sequences=True),        
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.LSTM(units=100),
    tf.keras.layers.Dense(units=total_words, activation='softmax')                        
])

model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())

# LambdaCallback : Lambda 함수를 작성하여 생성자에 넘기는 방식을 사용 
# fit() 할 때 학습 도중 train data predict 되는 내용을 확인해 가며 작업하고 싶을 때 사용할 수도 있다.
from keras.utils import pad_sequences

def testmodel_func(epoch, logs):
    if epoch %  5 != 0 and epoch != 49:
        return

    test_sentence = train_text[0]
    next_words = 100
    for _ in range(next_words):
        test_text_X = test_sentence.split(' ')[-seq_length:]
        test_text_X = np.array([word2idx[c] if c in word2idx else word2idx['UNK'] for c in test_text_X])
        test_text_X = pad_sequences([test_text_X], maxlen=seq_length, padding='pre', value=word2idx['UNK'])
        output_idx = np.argmax(model.predict(test_text_X)[0])
        test_sentence += ' ' + idx2word[output_idx]
    print()
    print(test_sentence)
    print()
  
testmodel_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=testmodel_func)

# steps_per_epoch : epoch당 사용한  strep 수를 지정, 예를 들어 총 45개의 sample이 있고 배치사이즈가 3이라면 15 step으로 지정
# 훈련데이터가 무한크기인 경우 유용하다.
history = model.fit(train_dataset.repeat(), epochs=50, steps_per_epoch=steps_per_epoch, 
                    callbacks=[testmodel_cb], verbose=2) # 무한반복

print(history.history['loss'][-1])
print(history.history['acc'][-1])
model.save('nlp7model.hdf5')

del model

from keras.models import load_model
model = load_model('nlp7model.hdf5')

# 임의의 문장을 사용해 문자 생성
test_sentence='수동이는 지리산에 구천이가 있다는 뜬소문을 생각 안 할래야 안  할 수가 없었다.'
next_words = 500

for _ in range(next_words):
    test_text_X = test_sentence.split(' ')[-seq_length:]
    test_text_X = np.array([word2idx[c] if c in word2idx else word2idx['UNK'] for c in test_text_X])
    test_text_X = pad_sequences([test_text_X], maxlen=seq_length, padding='pre', value=word2idx['UNK'])
    output_idx = np.argmax(model.predict(test_text_X)[0])
    test_sentence += ' ' + idx2word[output_idx]
print()
print(test_sentence)
print()