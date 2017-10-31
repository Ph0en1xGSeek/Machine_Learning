import requests
from bs4 import BeautifulSoup
import json
import re

# comments = requests.get('http://comment5.news.sina.com.cn/page/info?version=1&format=js&\
# channel=gn&newsid=comos-fynffnz3826906&group=&compress=0&ie=utf-8&oe=utf-8&\
# page=1&page_size=20')
#
#
# comments = comments.text.strip('var data=')
# jd = json.loads(comments)
# commentCount = jd['result']['count']['total']
# print(commentCount)
#
# newsurl = 'http://news.sina.com.cn/o/2017-10-31/doc-ifynffnz3826906.shtml'
# newsId = newsurl.split('/')[-1].lstrip('doc-i').rstrip('.shtml')
# print(newsId)
#
# #another version
# m = re.search('doc-i(.*).shtml', newsurl)
# print(m.group(1))


commentUrl = 'http://comment5.news.sina.com.cn/page/info?version=1&format=js&\
channel=gn&newsid=comos-{}&group=&compress=0&ie=utf-8&oe=utf-8&\
page=1&page_size=20'

# print(commentUrl.format(newsId))

def getCommentCount(newsurl):
    m = re.search('doc-i(.*).shtml', newsurl)
    newsId = m.group(1)
    comments = requests.get(commentUrl.format(newsId))
    comments = comments.text.strip('var data=')
    jd = json.loads(comments)
    return jd['result']['count']['total']

if __name__ == '__main__':
    newsurl = 'http://news.sina.com.cn/o/2017-10-31/doc-ifynffnz3826906.shtml'
    print(getCommentCount(newsurl))