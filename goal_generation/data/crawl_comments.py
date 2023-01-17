import time, random, json, time
from bs4 import BeautifulSoup
import requests
from pprint import pprint
from pywikibot import pagegenerators as pg
from selenium.webdriver import Chrome
from selenium_recaptcha import Recaptcha_Solver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests, csv, re
from lxml import etree
import pandas as pd
import numpy as np

EMOJI = re.compile("["
                u"\u00A9" 
                u"\u00AE" 
                u"\u203C" 
                u"\u2049" 
                u"\u20E3" 
                u"\u2122" 
                u"\u2139" 
                u"\u231A"
                u"\u2060" 
                u"\u231B" 
                u"\u2328" 
                u"\u23CF" 
                u"\u24C2" 
                u"\u25AA" 
                u"\u25AB" 
                u"\u25B6" 
                u"\u25C0" 
                u"\u2934" 
                u"\u2935"
                u"\u3030" 
                u"\u303D" 
                u"\u3297" 
                u"\u3299"
                u"\uFFFD"
                u"\u23E9-\u23F3" 
                u"\u23F8-\u23FA"
                u"\u25FB-\u25FE" 
                u"\u2600-\u27EF"  
                u"\u2B00-\u2BFF"                 
                u"\u2194-\u2199" 
                u"\u21A9-\u21AA"  
                u"\U0001F000-\U0001F02F" 
                u"\U0001F0A0-\U0001F0FF" 
                u"\U0001F100-\U0001F64F" 
                u"\U0001F680-\U0001F6FF" 
                u"\U0001F910-\U0001F96B" 
                u"\U0001F980-\U0001F9E0"
                        "]+", re.UNICODE)

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
        num = random.randint(0, 1)
        df.at[i, 'participant'] = '無' if num < 1 else '、'.join([s.replace('\n', '') for s in random.choices(usr_list, k=num)])
        df.at[i, 'event_content'] = '，'.join(df.at[i, 'event_content'].split('，')[:2]) if len(df.at[i, 'event_content'])>50 else df.at[i, 'event_content']
    df.to_csv(filename, encoding='utf-8-sig', index=False)
    
def get_comment_to_email():
    data=[['subject', 'content', 'copy_recipient', 'recipient', 'sender']]    
    ptt_gen = ptt_crawler()
    with open('user_list.txt', 'r', encoding = 'utf-8') as f:
        usr_list = f.readlines()
    try:
        while True:
            subject, content = next(ptt_gen)
            num = random.randint(0, 1)
            cp_r = '無' if num < 1 else '、'.join([s.replace('\n', '') for s in random.choices(usr_list, k=num)])
            rp = random.choice(usr_list).replace('\n', '')            
            sender = random.choice(usr_list).replace('\n', '')
            print([subject, EMOJI.sub(r'', content), cp_r, rp, sender])
            data.append([subject, EMOJI.sub(r'', content), cp_r, rp, sender])
    except:
        print(data)
        list_rows = np.array(data)
        np.savetxt("csv/mail_entityies.csv", list_rows, delimiter =",",fmt ='% s', encoding='utf-8-sig')

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
    page = int(next_url[1]["href"][-10:-5])
    
    for i in range(page-150, page):
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
                yield [title, '。'.join(article.split('。')[:2])]
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
    main_contents = soup.find(id="main-content")
    metas = main_contents.select('div.article-metaline')
    title = ''
    if metas:
        title = metas[1].select('span.article-meta-value')[0].string if metas[1].select('span.article-meta-value')[0] else title
    #articles = main_content.select('span.f2')
    #print(title)
    main_cons = []
    for main_content in main_contents.find_all(text=True, recursive=False)[0].split('\n'):
        if len(main_content.strip().replace(' ', ''))>1:main_cons.append(main_content.strip())
    article = re.sub(r"( +)", '', "，".join(main_cons))
    
    data = {
        'article_title': title,
        'article': article
    }
    #print(data, '\n')
    return [title.replace(',', '，').replace('[問卦]', '').replace('Re: ', '').replace('Fw: ', '').replace('[新聞]', ''), re.sub(r"[ -- a-zA-Z0-9]", '', article)]

if __name__ == '__main__':
    PTT_URL = 'https://www.ptt.cc'
    filename = 'ptt_article.csv'
    GAMER_URL = 'https://haha.gamer.com.tw/?room=hot'

    #ptt_crawler(filename)
    #get_user_list()
    get_comment_to_email()
    #calendar('csv/events.csv')
