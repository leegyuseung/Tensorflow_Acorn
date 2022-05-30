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
