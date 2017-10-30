import requests
from bs4 import BeautifulSoup
from datetime import datetime

res = requests.get('http://news.sina.com.cn/c/nd/2017-10-30/doc-ifynfrfn0452819.shtml')
res.encoding = 'utf-8'

soup = BeautifulSoup(res.text, 'html.parser')

title = soup.select('#artibodyTitle')[0].text
print(title)

timeSource = soup.select('.time-source')

timeStr = timeSource[0].contents[0].strip()# str type time
time = datetime.strptime(timeStr, '%Y年%m月%d日%H:%M')

source = timeSource[0].select('span span a')[0].text
print(time)
print(source)

article = []
contain = soup.select('#artibody p')[:-1] # remove the last one
for p in contain:
    article.append(p.text.strip())

article = '\n'.join(article)#join the article by \n


#simple version
article2 = '\n'.join([p.text.strip() for p in contain])



print(article2)
