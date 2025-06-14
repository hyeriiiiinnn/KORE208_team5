import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

stopwords = set(['후보', '대선', '대통령', '정치', '선거', '민주당', '한덕수', '한동훈', '국민', '이재명', '김문수', '이준석', '있', '하', '되', '같', '이런', '그런', '같', '대하', '이날', '아니', '기자', '더불어민주당', '국민의힘', '개혁신당', '민주노동당'])

# ▶ "단어(품사)" 형태에서 단어만 추출
def extract_tokens(tagged_text, allowed_tags={'NNG', 'NNP'}):
    tokens = []
    for item in tagged_text.split():
        if '(' in item and ')' in item:
            try:
                word, tag = item.rsplit('(', 1)
                tag = tag[:-1]
                if tag in allowed_tags and word not in stopwords:
                    tokens.append(word)
            except:
                continue
    return tokens

# ▶ 파일 로딩
def load_token_list(path):
    df = pd.read_csv(path)
    all_tokens = []
    for text in df['tagged']:
        all_tokens += extract_tokens(text)
    return all_tokens

# ▶ TF 계산
def get_top_tf(tokens, top_n=50):
    counter = Counter(tokens)
    return counter.most_common(top_n)

# ▶ TF-IDF 계산
def get_top_tfidf(corpus_dict, target_key, top_n=50):
    corpus = [' '.join(tokens) for tokens in corpus_dict.values()]
    keys = list(corpus_dict.keys())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    idx = keys.index(target_key)
    row = tfidf_matrix[idx].toarray().flatten()
    tfidf_scores = {feature_names[i]: row[i] for i in range(len(row))}
    top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_words

# ▶ 일치율 계산
def compute_overlap(tf_words, tfidf_words):
    tf_set = set(w for w, _ in tf_words)
    tfidf_set = set(w for w, _ in tfidf_words)
    return len(tf_set & tfidf_set) / len(tfidf_set)

# ▶ 경로
paths = {
    '이재명': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/lee_jaemyung_tagged_filtered.csv',
    '김문수': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/kim_moonsu_tagged_filtered.csv',
    '이준석': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/lee_junseok_tagged_filtered_ver2.csv'
}

# ▶ 전체 실행
tokens_dict = {k: load_token_list(v) for k, v in paths.items()}
results = []

for name in paths.keys():
    tf_top = get_top_tf(tokens_dict[name])
    tfidf_top = get_top_tfidf(tokens_dict, name)
    overlap = compute_overlap(tf_top, tfidf_top)

    tf_words = [w for w, _ in tf_top]
    tfidf_words = [w for w, _ in tfidf_top]

    print(f"\n[{name}]")
    print(f"TF-TFIDF 상위 50 단어 일치율: {overlap:.2%}")
    print(f"TF 상위 단어: {tf_words}")
    print(f"TF-IDF 상위 단어: {tfidf_words}")

    results.append({
        '정치인': name,
        '일치율(%)': round(overlap * 100, 2),
        'TF 상위 단어': ', '.join(tf_words),
        'TF-IDF 상위 단어': ', '.join(tfidf_words)
    })

# ▶ 저장
df_result = pd.DataFrame(results)
df_result.to_csv("TF_TFIDF_일치율_및_상위단어.csv", index=False, encoding='utf-8-sig')


# 코사인 유사도 분석 추가
corpus = {name: ' '.join(tokens) for name, tokens in tokens_dict.items()}

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus.values())
names = list(corpus.keys())

similarity_matrix = cosine_similarity(tfidf_matrix)
df_sim = pd.DataFrame(similarity_matrix, index=names, columns=names)

print("\n정치인 간 TF-IDF 코사인 유사도:")
print(df_sim.round(4))

# 코사인 유사도 결과 저장
df_sim.to_csv("정치인_코사인_유사도.csv", encoding='utf-8-sig')