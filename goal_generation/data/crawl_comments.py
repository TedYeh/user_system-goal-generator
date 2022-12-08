import time, random, json
from bs4 import BeautifulSoup
import requests
from pprint import pprint
from pywikibot import pagegenerators as pg
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests, csv, re
from lxml import etree
import pandas as pd
import numpy as np


def get_youtube_comment():
    data=[['users', 'App', 'Comments']]
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                        "]+", re.UNICODE)

    with Chrome(executable_path=r'C:\Program Files\chromedriver.exe') as driver:
        
        with open('instant_messaging.txt', 'r', encoding = 'utf-8') as f:
            im_list = f.readlines()

        url_list = ['https://youtu.be/xmjHnr6u9BM', 'https://youtu.be/cTIWfkjgerE', 'https://youtu.be/n99n_QsBi_o']#['https://youtu.be/xbV_IoXo5tc', "https://youtu.be/aTMnqsvOo74"]    
        wait = WebDriverWait(driver,150)
        for url in url_list:
            driver.get(url)

            for item in range(20): 
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(5)
        users = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#author-text")))
        comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))
        for user, comment in zip(users, comments):
            data.append([user.text, random.choice(im_list).replace('\n', ''),comment.text.replace('\n', '')])
        
    list_rows = np.array(data)
    np.savetxt("csv/message_entityies.csv", list_rows, delimiter =",",fmt ='% s', encoding='utf-8-sig')

def calendar(filename):
    with open('user_list.txt', 'r', encoding = 'utf-8') as f:
        usr_list = f.readlines()
    df = pd.read_csv(filename)
    for i in range(len(df['participant'])):
        num = random.randint(0, 3)
        df.at[i, 'participant'] = '' if num < 1 else '、'.join([s.replace('\n', '') for s in random.choices(usr_list, k=num)])
    df.to_csv(filename, encoding='utf-8-sig', index=False)
    
def get_comment_to_email():
    data=[['subject', 'content', 'copy_recipient', 'recipient', 'sender']]    
    ptt_gen = ptt_crawler()
    with open('user_list.txt', 'r', encoding = 'utf-8') as f:
        usr_list = f.readlines()
    try:
        while True:
            subject, content = next(ptt_gen)
            cp_r = '、'.join([s.replace('\n', '') for s in random.choices(usr_list, k=random.randint(0, 3))])
            rp = random.choice(usr_list).replace('\n', '')
            sender = random.choice(usr_list).replace('\n', '')
            data.append([subject, content, cp_r, rp, sender])
    except:
        list_rows = np.array(data)
        np.savetxt("csv/mail_entityies.csv", list_rows, delimiter =",",fmt ='% s', encoding='utf-8-sig')

def get_user_list():
    data = []
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                        "]+", re.UNICODE)

    with Chrome(executable_path=r'C:\Program Files\chromedriver.exe') as driver:

        url_list = ['https://youtu.be/9H9wT-zoMuA', 'https://youtu.be/OE9mcx_iJrE', 'https://youtu.be/RMRprLmAWw4', 'https://youtu.be/M8-c90KQQCI', 'https://youtu.be/hrZ8AETuA7I']#['https://youtu.be/xbV_IoXo5tc', "https://youtu.be/aTMnqsvOo74"]    
        wait = WebDriverWait(driver,60)
        for url in url_list:
            driver.get(url)

            for item in range(30): 
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(3)
        users = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#author-text")))
        comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))
        for user, comment in zip(users, comments):
            data.append(user.text)

    with open('user_list.txt', 'w', encoding='utf-8-sig') as txt: txt.write('\n'.join(data))

def ptt_crawler(filename=''):
    data=[]
    resp = requests.get(
        url = PTT_URL + '/bbs/Gossiping/index.html',
        cookies = {'over18': '1'},
        verify = True,
        timeout = 3
    )
    if resp.status_code != 200:
        print('Invalid url:', resp.url)

    soup = BeautifulSoup(resp.text,"html.parser")
    next_url = soup.select("div.btn-group.btn-group-paging a")
    page = next_url[1]["href"][-9:-5]
    
    for i in range(39000, 39115):
        resp = requests.get(
            url = PTT_URL + '/bbs/Gossiping/index' + str(i) + '.html',
            cookies = {'over18': '1'}
        )
        if resp.status_code != 200:
            print('Invalid url:', resp.url)
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        divs = soup.find_all("div", "r-ent")        
        for div in divs:
            try:
                href = div.find('a')['href']
                link = PTT_URL + href
                article_id = re.sub('\.html', '', href.split('/')[-1])
                #print(link, article_id + '\n')
                title, article = parse_article(link, article_id)
                if len(article) <= 5: continue
                yield [title, article]
                #data.append([title, article])
            except:
                pass

    #list_rows = np.array(data)
    #np.savetxt(f"csv/{filename}", list_rows, delimiter =",", fmt ='% s', encoding='utf-8-sig')

def parse_article(link, article_id):
    resp = requests.get(
        url = link,
        cookies = {'over18': '1'},
        verify = True,
        timeout = 3
    )
    if resp.status_code != 200:
        print('Invalid url:', resp.url)
        return json.dumps({"error": "invalid url"}, sort_keys=True, ensure_ascii=False)

    soup = BeautifulSoup(resp.text, 'html.parser')
    main_content = soup.find(id="main-content")
    metas = main_content.select('div.article-metaline')
    title = ''
    if metas:
        title = metas[1].select('span.article-meta-value')[0].string if metas[1].select('span.article-meta-value')[0] else title
    #articles = main_content.select('span.f2')
    #print(title)
    article = " ".join(main_content.find_all(text=True, recursive=False)[0].split('\n')[1:])
    
    data = {
        'article_title': title,
        'article': article
    }
    #print(data, '\n')
    return [title.replace(',', '，').replace('[問卦]', '').replace('Re: ', ''), article.replace(',', '，')]

if __name__ == '__main__':
    PTT_URL = 'https://www.ptt.cc'
    filename = 'ptt_article.csv'
    #ptt_crawler(filename)
    #get_user_list()
    #get_comment_to_email()
    #get_user_list()
    calendar('csv/events.csv')