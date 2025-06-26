import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta
import re
import time

def fetch_url(url, retries=3, delay=0.3):
    """
    입력받은 url에 대해 fetch 요청
    기본값으로는 0.3초 간격으로 최대 3회 재시도
    만약 attempt < retries라면 return None
    """
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}. Retrying ({attempt + 1}/{retries})...")
            attempt += 1
            time.sleep(delay)
    return None

def parse_crawled(article_url):
    article_response = fetch_url(article_url)

    if article_response is None:
        print(f"Failed to fetch article {article_url} after multiple attempts.")
        return None

    article_soup = BeautifulSoup(article_response.content, 'html.parser')

    try:
        title = article_soup.find('h2', class_='media_end_head_headline').text.strip()
    except AttributeError:
        title = '정보를 찾을 수 없음'

    try:
        date = article_soup.find('span', class_='media_end_head_info_datestamp_time _ARTICLE_DATE_TIME').text.strip()
    except AttributeError:
        date = '정보를 찾을 수 없음'

    try:
        content = article_soup.find('article', class_='_article_content').text.strip()
    except AttributeError:
        content = '정보를 찾을 수 없음'

    article_data = {
        'title': title,
        'date': date,
        'content': content,
        'source': article_url
    }

    return article_data