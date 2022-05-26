# 단어의 빈도수를 이용해 문서의 특징 추출
# BOW(Bag of Words) : 문서가 가지는 모든 단어, 문맥, 순서를 무시하고 단어에 대해 빈도 수를 부여해 벡터를 생성


# CountVectorizer : 단순하게 텍스트에서 단위별 등장횟수를 카운팅하여 수치벡터(BOW)화
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

contents = ['How to formt my hard disk', 'Hard disk format format problems']

count_vec = CountVectorizer(analyzer='word', min_df=1)
tran = count_vec.fit_transform(raw_documents=contents)
print(tran)
print(count_vec.get_feature_names())
print(tran.toarray())

print('--------------')
# TfidfVectorizer : 문서에서 자주 나오는 단어에 대해 가중치를 높게 부여하나,
# 전체 문성의 모든 문단에서 자주 등장하는 단어에 대해서는 패널티를 주는 방식으로 값을 부여.
tfidf_vec = TfidfVectorizer(analyzer = 'word', min_df = 1)
tran_idf = tfidf_vec.fit_transform(raw_documents=contents)
print(tran_idf)
print(tfidf_vec.get_feature_names())
print(tran_idf.toarray())

