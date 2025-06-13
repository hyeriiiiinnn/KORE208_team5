# 1. 라이브러리 임포트
import json
import re
import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm

# 2. JSON 로딩
with open("/Users/matthew941/Desktop/KORE208_team5/3. 정치인별 빈도, 워드임베딩/2.김문수/김문수.json", "r", encoding="utf-8") as f:
    data = json.load(f)
articles = data["김문수"]  # 리스트

# 3. 텍스트 전처리 함수
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'["“”‘’\'!?…·,;:\(\)\[\]\{\}]', '', text)
    text = re.sub(r'[^\w\s가-힣]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 4. 형태소 분석기 초기화 및 사용자 단어 등록
kiwi = Kiwi()
user_words = ['개혁신당', '더불어민주당', '국민의힘', '최고득표율', '이 후보', '인공지능',"직권남용",'양자대결',"김 후보","출구조사","선거법","더중플","권영국"]
for word in user_words:
    kiwi.add_user_word(word, 'NNP', 0)

# 5. 불용어 및 제거 품사 정의
stopwords = set([
    "것", "수", "등", "더", "때", "자신", "이번", "이후", "위해", "관련",
    "의견", "최근", "대해", "내용", "경우", "부분", "정도", "현재", "상황",
    "사실", "또한", "통해", "가장", "대한", "전혀", "하나", "바로", "계속",
    "여러", "많은", "이미", "이", "그", "저", "우리", "그것", "이것"
])
stop_tags = {
    'JKS', 'JKC', 'JKG', 'JKB', 'JKO', 'JKV', 'JKQ',   # 조사
    'JX', 'JC',                                # 보조사, 접속조사
    'NNB',                                     # 의존명사
    'NR', 'SN',                                # 수사, 숫자
    'SF', 'SP', 'SS', 'SE', 'SO',              # 구두점
    'EP', 'EF', 'EC', 'ETN', 'ETM',            # 어미
    'XPN', 'XSN', 'XSV', 'XSA',                # 접사
    'IC',                                      # 감탄사
    'SF', 'SE', 'SS', 'SP', 'SO',              # 기호
    'VSV', 'NR'
}

# 6. 기사 단위 전처리 및 분석
titles, dates, originals, cleaneds, taggeds = [], [], [], [], []

for article in tqdm(articles):
    title = article.get("title", "")
    date = article.get("date", "")
    original = article.get("content", "")

    cleaned = clean_text(original)

    analyzed = kiwi.analyze(cleaned)[0][0]
    tagged_tokens = [
        f"{token.form}({token.tag})"
        for token in analyzed
        if token.tag not in stop_tags and token.form not in stopwords
    ]
    tagged = " ".join(tagged_tokens)

    titles.append(title)
    dates.append(date)
    originals.append(original)
    cleaneds.append(cleaned)
    taggeds.append(tagged)

# 7. DataFrame 생성
df = pd.DataFrame({
    "title": titles,
    "date": dates,
    "original": originals,
    "cleaned": cleaneds,
    "tagged": taggeds
})

# 8. CSV 저장
df.to_csv("김문수_tagged_filtered_ver2.csv", index=False, encoding="utf-8-sig")

