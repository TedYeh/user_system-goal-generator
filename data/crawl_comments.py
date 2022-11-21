import time, random
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests, csv, re
from lxml import etree
import pandas as pd
import numpy as np


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
        
    wait = WebDriverWait(driver,15)
    driver.get("https://www.youtube.com/watch?v=BLfE6a5bAX4")

    for item in range(3): 
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(2)
    users = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#author-text")))
    comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))
    for user, comment in zip(users, comments):
        data.append([user.text, random.choice(im_list).replace('\n', ''),comment.text.replace('\n', '')])
    # style-scope ytd-comment-renderer
    list_rows = np.array(data)
    np.savetxt("message_entityies.csv", list_rows, delimiter =",",fmt ='% s', encoding='utf-8-sig')
    