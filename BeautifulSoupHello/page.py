import requests
import json
import newsDetail
import pandas

# res = requests.get('http://api.roll.news.sina.com.cn/zt_list?channel=news&cat_1=gnxw&cat_2==gdxw1||=gatxw||=zs-pl||=mtjj&level==1||=2&show_ext=1&show_all=1&show_num=22&tag=1&format=json&page=4&callback=newsloadercallback&_=1509456814169')
# # print(res.text)
#
# restext = res.text.lstrip('  newsloadercallback(').rstrip(');')
# jd = json.loads(restext)
#
# for ent in jd['result']['data']:
#     print(ent['url'])

def parseListLinks(url):
    res = requests.get(url)
    newsdetails = []
    restext = res.text.lstrip('  newsloadercallback(').rstrip(');')
    jd = json.loads(restext)

    for ent in jd['result']['data']:
        newsdetails.append(newsDetail.getNewsDetail(ent['url']))
    return newsdetails


if __name__ == '__main__':
    news_total = []
    url = 'http://api.roll.news.sina.com.cn/zt_list?channel=news&cat_1=gnxw&cat_2==gdxw1||=gatxw||=zs-pl||=mtjj&level==1||=2&show_ext=1&show_all=1&show_num=22&tag=1&format=json&page={}&callback=newsloadercallback&_=1509456814169'
    for i in range(4, 5):
        newsurl = url.format(i)
        newsarry = parseListLinks(newsurl)
        news_total.extend(newsarry)
    df = pandas.DataFrame(news_total)
    # df.to_excel('news.xlsx')
    import sqlite3
    with sqlite3.connect('news.sqlite') as db:
        df.to_sql('news', con = db)

    with sqlite3.connect('news.sqlite') as db:
        df2 = pandas.read_sql_query('select * from news', con = db)
    print(df2)
