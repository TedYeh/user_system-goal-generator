def default_state():
    state = dict(user_action=[], 
                 system_action=[], 
                 belief_state={}, 
                 cur_domain=None, 
                 request_slots=[], 
                 terminated=False,
                 history=[])
    state['belief_state'] = {
        'Calendar': {
              '活動名稱': '',
              '活動時間': '',
              '參加者': '',
              '是否全天': '',
              '活動內容': '',
              '活動地點': ''
            },
            'Gmail': {
              '收件者': '',
              '寄件者': '',
              '郵件主旨': '',
              '信件內容': '',
              '副本收件者': '',
              '密件副本收件者': ''
            },
            'Message': {
              '收件者': '',
              '應用程式': '',
              '傳送內容': ''
            }
    }
    return state
