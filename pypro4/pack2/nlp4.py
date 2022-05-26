# 네이버의 영화 정보 사용
# 5편의 영화 리뷰를 읽어 형태소 분석 후 CountVectorizer, TfidfVectorizer 적용
# 각 영화 간의 코사인유사도 확인

import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def movie_scrap_func(url):
    result = []
    for p in range(1,6): #5 페이지 리뷰
        r = requests.get(url + "&page="+ str(p))
        # print(r) <Response [200]>
        soup = BeautifulSoup(r.content, 'lxml', from_encoding='ms949')
        # print(soup)
        title = soup.find_all("td", {"class":"title"})
        # print(title[1].text)
        sub_result = []
        for i in range(len(title)):
            sub_result.append(title[i].text
                    .replace("\r","")
                    .replace("\n","")
                    .replace("\t","")
                    .replace("ㅋ","")                     
                    .replace("ㅠ","")
                    .replace("-","")
                    .replace("신고","")
                    .replace("별점","")
                    .replace("총","")
                    .replace("점","")
                    .replace("중","")
                    .replace("범죄도시2","")
                    .replace("범죄도시","")
                    .replace("그대가 조국","")
                    .replace("닥터 스트레이지","")
                    .replace("연애 빠진 로맨스","")
                    .replace("사냥의 시간",""))    
        result = result + sub_result
    return("".join(result))

# 범죄도시2, 그대가 조국, 닥터스트레이지, 연애빠진로맨스, 사냥의시간  
city2 = movie_scrap_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=192608&target=after")
jokuk = movie_scrap_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=216100&target=after")
doctor = movie_scrap_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=182016&target=after")
romance = movie_scrap_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=196809&target=after")
sigan = movie_scrap_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=174808&target=after")
movies = [city2, jokuk, doctor, romance, sigan]
# print(movies)

print('------------')
okt = Okt()

def word_separate(movies):
    result = []
    for movie in movies:
        words = okt.pos(movie)
        one_result = []
        for word in words:
            if word[1] in ['None','Adjective'] and len(word[0]) >=2 :
                one_result.append(word[0])
        result.append(" ".join(one_result))
    return result

word_list = word_separate(movies)
print(word_list)

print('-----CountVectorizer, TfidfVectorizer-----')
# CountVectorizer
count = CountVectorizer(min_df = 2) # 빈도수 2이하는 무시한다. // 문서집합으로 단어의 술를 세어 BOW 벡터를 만듦 
count_dtm = count.fit_transform(word_list).toarray()
print(count_dtm)
# count_dtm을 DataFrame에 저장
cou_dtm_df = pd.DataFrame(count_dtm,
                          columns = count.get_feature_names(),
                          index = ['city2', 'jokuk', 'doctor', 'romance', 'sigan'])
print(cou_dtm_df)

# TfidfVectorizer : 단어의 빈도수에 추가적으로 단어의 중요성까지 가중치로 표현
idf_maker = TfidfVectorizer(min_df=2)
tfidf_dtm = idf_maker.fit_transform(word_list).toarray()
print(tfidf_dtm)

tfidf_dtm_df = pd.DataFrame(tfidf_dtm,
                            columns = idf_maker.get_feature_names(),
                            index = ['city2', 'jokuk', 'doctor', 'romance', 'sigan'])
print(tfidf_dtm_df)

print('각 영화 간에 유사도 확인 : 코사인 유사도 공식 사용')
def cosine_function(doc1, doc2):
    bunja = sum(doc1 * doc2) 
    bunmo = (sum(doc1 ** 2) * sum(doc2 ** 2)) ** 0.5
    return bunja / bunmo

res = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        res[i, j] = cosine_function(tfidf_dtm_df.iloc[i], tfidf_dtm_df.iloc[j].values)
        
print(res)

df = pd.DataFrame(res, index = ['city2', 'jokuk', 'doctor', 'romance', 'sigan'],
                  columns = ['city2', 'jokuk', 'doctor', 'romance', 'sigan'])
print(df)