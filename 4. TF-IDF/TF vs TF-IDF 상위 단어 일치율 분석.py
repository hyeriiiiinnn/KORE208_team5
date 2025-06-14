import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ▶ 불용어
stopwords = set(['후보', '대선', '대통령', '정치', '국민', '이재명', '김문수', '이준석', '있', '하', '되', '같'])

# ▶ "단어(품사)" 형태에서 단어만 추출
def extract_tokens(tagged_text, allowed_tags={'NNG', 'NNP'}):
    tokens = []
    for item in tagged_text.split():
        if '(' in item and ')' in item:
            try:
                word, tag = item.rsplit('(', 1)
                tag = tag[:-1]  # ')' 제거
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
    return counter.most_common(top_n)  # 리스트[(단어, 빈도)]

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
    return top_words  # 리스트[(단어, TF-IDF)]

# ▶ TF-TFIDF 일치율 계산
def compute_overlap(tf_list, tfidf_list):
    tf_set = set([w for w, _ in tf_list])
    tfidf_set = set([w for w, _ in tfidf_list])
    return len(tf_set & tfidf_set) / len(tfidf_set)

# ▶ 경로 정의
paths = {
    '이재명': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/lee_jaemyung_tagged_filtered.csv',
    '김문수': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/kim_moonsu_tagged_filtered.csv',
    '이준석': 'https://raw.githubusercontent.com/hyeriiiiinnn/KORE208_team5/main/4.%20TF-IDF/lee_junseok_tagged_filtered_ver2.csv'
}

# ▶ 토큰 추출
tokens_dict = {k: load_token_list(v) for k, v in paths.items()}

results = []

for name in paths.keys():
    tf_list = get_top_tf(tokens_dict[name])
    tfidf_list = get_top_tfidf(tokens_dict, name)
    overlap = compute_overlap(tf_list, tfidf_list)

    print(f"\n▶ [{name}]")
    print(f"TF-TFIDF 상위 50 단어 일치율: {overlap:.2%}")
    print("상위 50개 TF 단어:")
    print(', '.join([f"{w}({c})" for w, c in tf_list]))
    print("\n상위 50개 TF-IDF 단어:")
    print(', '.join([f"{w}({round(score, 4)})" for w, score in tfidf_list]))

    results.append({'정치인': name, '일치율(%)': round(overlap * 100, 2)})

# ▶ 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv("C:/Users/master/Documents/KakaoTalk Downloads/TF_TFIDF_일치율.csv", index=False, encoding='utf-8-sig')
