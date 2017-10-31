import requests
from bs4 import BeautifulSoup
from datetime import datetime
import commentList

# res = requests.get('http://news.sina.com.cn/o/2017-10-31/doc-ifynffnz3826906.shtml')
# res.encoding = 'utf-8'
#
# soup = BeautifulSoup(res.text, 'html.parser')
#
# title = soup.select('#artibodyTitle')[0].text
# print(title)
#
# timeSource = soup.select('.time-source')
#
# timeStr = timeSource[0].contents[0].strip()# str type time
# time = datetime.strptime(timeStr, '%Y年%m月%d日%H:%M')
#
# source = timeSource[0].select('span span a')[0].text
# print(time)
# print(source)
#
# article = []
# contain = soup.select('#artibody p')[:-1] # remove the last one
# for p in contain:
#     article.append(p.text.strip())
#
# article = '\n'.join(article)#join the article by \n
#
#
# #simple version
# article2 = '\n'.join([p.text.strip() for p in contain])
#
#
#
# print(article2)
#
# editor = soup.select('.article-editor')[0].text
#
# commentCount = soup.select('#commentCount1')

def getNewsDetail(newsurl):
    result = {}
    res = requests.get(newsurl)
    res.encoding = 'utf-8'

    soup = BeautifulSoup(res.text, 'html.parser')
    result['title'] = soup.select('#artibodyTitle')[0].text
    timeSource = soup.select('.time-source')

    timeStr = timeSource[0].contents[0].strip()  # str type time
    result['dt'] = datetime.strptime(timeStr, '%Y年%m月%d日%H:%M')
    result['source'] = timeSource[0].select('span span a')[0].text
    result['article'] = '\n'.join([p.text.strip() for p in soup.select('#artibody p')[:-1]])
    result['editor'] = soup.select('.article-editor')[0].text.strip('责任编辑：')
    result['comments'] = commentList.getCommentCount(newsurl)
    return result

if __name__ == '__main__':
    newsurl = 'http://news.sina.com.cn/o/2017-10-31/doc-ifynffnz3826906.shtml'
    res = getNewsDetail(newsurl)
    print(res)