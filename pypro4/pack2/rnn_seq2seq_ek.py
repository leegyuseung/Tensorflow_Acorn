# seq2seq로 영어를 한국어로 번역하는 번역 모델 생성
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# data
data_path = '../testdata/eng_kor.txt'
lines = open(data_path, mode='r', encoding='utf-8').read().split('\n')
print(lines[:10])
print(len(lines))

# 샘플 나누기
num_samples = 10000
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for line in lines[:min(num_samples, len(lines))]:
    input_text, target_text = line.split('\t')     # Go. 가.
    # print(input_text)
    # print(target_text)
    target_text ='\t' + target_text + '\n'  # \t:<sos> \n:<eos>로 사용
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        input_characters.add(char)

    for char in target_text:
        target_characters.add(char)

# 정렬
input_characters = sorted(input_characters)
target_characters = sorted(target_characters)        
print(input_texts[:-10])
print(target_texts[:-10])
print(input_characters)
print(target_characters)

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('샘플 수:', len(input_texts))
print('영어 글자 수:', num_encoder_tokens)
print('한글 글자 수:', num_decoder_tokens)
print('영어 중 가장 긴 단어 글자 수:', max_encoder_seq_length)
print('한글 중 가장 긴 단어 글자 수:', max_decoder_seq_length)

# 글자 집합에 글자 단위로 저장된 각 글자에 대해 index 부여
input_token_index = dict([(char,i) for i, char in enumerate(input_characters)])
print(input_token_index)

target_token_index = dict([(char,i) for i, char in enumerate(target_characters)])
print(target_token_index)

# One-hot
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
print(encoder_input_data.shape) # (595, 15, 59)
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
print(decoder_input_data.shape) # (595, 18, 427)
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
print(decoder_target_data.shape) # (595, 18, 427)

# 0으로 채워진 벡터에 해당 글자가 있는 지점은 1을 기억
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # print(input_text, target_text)
    for t, char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[char]]=1

    for t, char in enumerate(target_text):
        # decoder_target_index가 encoder_input_text 보다 한 스텝 앞서 입력됨!
        decoder_input_data[i,t, target_token_index[char]]=1  
        if t > 0:
            # decoder_target_data는 한 타임 스텝 만큼 앞당겨지며 시작문자를 포함하지 않음 
            decoder_target_data[i, t-1, target_token_index[char]] = 1

# print(encoder_input_data[5]) # 데이터 있는 부분만 1로 채워진다.

# 네트워크 설계 - function api 사용
# 인코더 설계
latent_dim = 1024 # 인코딩 공간 차원

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

encoder_states = [state_h, state_c] # LSTM은 RNN과 달리 상태가 두 개. 은닉상태와 셀상태를 기억. 이 것이 바로 context vector

# 디코더의 첫 상태를 인코더의 은닉상태와 셀상태로 설정
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=latent_dim, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

batch_size=64
epochs = 50
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())

model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2)
model.save('s2s_ek.h5')

# seq2seq 번역기 동작
# 인코더 모델
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# 번역 동작 단계
# 1. 번역하고자 하는 입력문장이 인코더에 들어와서 은닉상태와 셀상태를 만든다.
# 2. 상태와 <sos>에 해당하는 '\t'를 디코더로 전달한다.
# 3. 디코더가 <eos>에 해당하는 '\n'이 나올 때까지 다음 문자를 예측하는 행동을 반복한다.

# 디코더 모델
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 문장의 다음 단어를 예측하기 위해서 초기상태를 이전 시점의 상태로 사용
decoder_states = [state_h, state_c]

# 인코더와 다르게 LSTM이 반환하는 은닉상태와 셀 상태를 버리지 않음
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                      outputs=[decoder_outputs]+decoder_states)

print(decoder_model.summary())

# 시퀀스를 다시 디코딩하는 역방향 조회 토큰 인덱스. 단어로 부터 인덱스를 얻는 것이 아니라 인덱스로부터 단어 얻기
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
print(reverse_input_char_index)
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
print(reverse_target_char_index)

def decode_seq_func(input_seq):
    # 입력으로 부터 인코더 상태를 얻음
    states_value = encoder_model.predict(input_seq)
    # print('states_value:', states_value) # [array([[ 2.2908259e-04,  2.2446758e-03, -1.9103340e-03, ...,
    
    target_seq = np.zeros((1,1,num_decoder_tokens)) # 길이가 1인 타겟 시퀀스 
    # 대상 시퀀스의 첫 번째 문자를 시작문자로 채움
    target_seq[0, 0, target_token_index['\t']] = 1. # <sns>에 해당하는 원핫벡터
    # print('target_seq:',target_seq) # [[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    
    # 시퀀스 배치에 대한 샘플링 반복 처리
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # 이전 시점 상태 state_value를 현재의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # 예측결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # print('sampled_token_index:',sampled_token_index)
        sampled_char = reverse_target_char_index[sampled_token_index]
        # print('sampled_char:',sampled_char)
        decoded_sentence += sampled_char # 현재 시점의 예측문자를 예측문자에 누적
        print('decode_sentence', decoded_sentence)

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length: # 최대길이 or <eos>를 만나면 종료
            _condition = True

        target_seq = np.zeros((1,1,num_decoder_tokens)) # <sos>에 해당하는 onehot벡터 생성
        target_seq[0,0,sampled_token_index] = 1
        states_value = [h,c] # states_value 갱신. 현 상태를 다음 시점의 상태로 사용
        print('states_value', states_value)

    return decoded_sentence

for seq_idx in range(5): #[3,67,8,9,10]
    # 디코딩을 위해 하나의 시퀀스 (학습 데이터)를 가져옴
    input_seq = encoder_input_data[seq_idx:seq_idx+1]
    results = decode_seq_func(input_seq)
    print('-------------------------')
    print('입력(영어):', input_texts[seq_idx])
    print('번역결과(한국어):', results)

