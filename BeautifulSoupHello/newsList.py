#coding:utf-8
import requests
from bs4 import BeautifulSoup

urls = 'http://news.sina.com.cn/china/'
# print(type(urls))
res = requests.get(urls)
res.encoding = 'utf-8'
# print(res.text)

soup = BeautifulSoup(res.text, 'html.parser')

# print(soup.text)

# header = soup.select('a')#类似CSS
# for h in header:
#     print(h.text)
#     print(h['href'])

#news list
for news in soup.select('.news-item'):
    if(len(news.select('h2')) > 0):
        h2 = news.select('h2')[0].text
        time = news.select('.time')[0].text
        a = news.select('a')[0]['href']
        print(time, h2, a)