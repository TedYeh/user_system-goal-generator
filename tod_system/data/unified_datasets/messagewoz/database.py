"""Modifed from https://github.com/thu-coai/CrossWOZ/blob/master/convlab2/util/crosswoz/dbquery.py"""
import json
import os
import re
from zipfile import ZipFile

from convlab.util.unified_datasets_util import (BaseDatabase,
                                                download_unified_datasets)

def time_to_str(times):
    if isinstance(times, str): return times
    elif not times[0]: return ''
    else:
        time_str = ''
        date1, t1 = times[0].split('T')
        d, h, m, amOrpm = date1[5:].replace('-', '/'), t1[:2]+'點', '' if int(t1[3:]) == 0 else t1[3:]+'分', '早上' if int(t1[:2]) < 12 else '下午'
        time_str += d + amOrpm + h + m    
        date2, t2 = times[1].split('T')
        d, h, m, amOrpm = '' if date2[5:].replace('-', '/')==date1[5:].replace('-', '/') else date2[5:].replace('-', '/')\
            , t2[:2]+'點', '' if int(t2[3:]) == 0 else t2[3:]+'分', '早上' if int(t2[:2]) < 12 else '下午'
        time_str += '到'+ d + amOrpm + h + m
        #print(times, time_str)
        return time_str

def contains(arr, s):
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))

class Database(BaseDatabase):
    def __init__(self):
        """extract data.zip and load the database."""
        data_path = download_unified_datasets('messagewoz', 'data.zip', os.path.dirname(os.path.abspath(__file__)))
        archive = ZipFile(data_path)
        self.domains = ['Calendar', 'Gmail', 'Message']
        domain2eng = {'Calendar': 'gcalendar', 'Gmail': 'gmail', 'Message': 'message'}
        self.data = {}
        for domain in self.domains[:-1]:
            with archive.open('data/{}_db.json'.format(domain2eng[domain])) as f:
                self.data[domain] = json.loads(f.read())
        
        self.schema = {
            'Calendar': {
              '活動名稱': { 'params': None },
              '活動時間': { 'params': None },
              '參加者': { 'type': 'multiple_in', 'params': None },
              '是否全天': { 'params': None },
              '活動內容': { 'params': None },
              '活動地點': { 'params': None }
            },
            'Gmail': {
              '收件者': { 'type': 'multiple_in', 'params': None },
              '寄件者': { 'params': None },
              '郵件主旨': { 'params': None },
              '信件內容': { 'params': None },
              '副本收件者': { 'type': 'multiple_in', 'params': None },
              '密件副本收件者': { 'type': 'multiple_in', 'params': None }
            },
            'Message': {
              '收件者': { 'type': 'multiple_in', 'params': None },
              '應用程式': { 'params': None },
              '傳送內容': { 'params': None }
            }
        }

    def query(self, domain: str, state: dict, topk: int) -> list:
        """
        return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state.
        query database using belief state, return list of entities, same format as database
        :param state: belief state of the format {domain: {slot: value}}
        :param domain: maintain by DST, current query domain
        :param topk: max number of entities
        :return: list of entities
        """
        if not domain or domain=='Message':
            return []
        cur_query_form = {}
        for slot, value in state[domain].items():
            if not value: continue
            cur_query_form[slot] = value
        #print('cur_query_form', cur_query_form, domain, state)
        cur_res = self.query_schema(field=domain, args=cur_query_form)
        res = cur_res

        return res[:topk]
    
    def query_schema(self, field, args):
        if not field in self.schema:
            raise Exception('Unknown field %s' % field)
        if not isinstance(args, dict):
            raise Exception('`args` must be dict')
        
        db = self.data.get(field)
        #print('db',db,)
        plan = self.schema[field]
        for key, value in args.items():
            if key == '活動時間':
                value = time_to_str(value)
            if not key in plan:
                raise Exception('Unknown key %s' % key)
            value_type = plan[key].get('type')
            if value_type == 'between':
                if not value[0] is None:
                    plan[key]['params'][0] = float(value[0])
                if not value[1] is None:
                    plan[key]['params'][1] = float(value[1])
            else:
                if not isinstance(value, str):
                    raise Exception('Value for `%s` must be string' % key)
                plan[key]['params'] = value

        def func3(item):
                        
            for key, val in args.items(): 
                if not item: continue 
                details = item[1]             
                #print(key, val, details)
                if isinstance(details, list):val = details[1].get(key)
                else:val = details.get(key)
                absence = val is None
                options = plan[key]
                if options.get('type') == 'between':
                    L = options['params'][0]
                    R = options['params'][1]
                    if not L is None:
                        if absence:
                            return False
                    else:
                        L = float('-inf')
                    if not R is None:
                        if absence:
                            return False
                    else:
                        R = float('inf')
                    if L > val or val > R:
                        return False
                elif options.get('type') == 'in':
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if contains(val, s):
                            return False
                elif options.get('type') == 'multiple_in':
                    s = options['params']
                    if not s is None and val:
                        if absence:
                            return False
                        sarr = list(filter(lambda t: bool(t), s.split(',')))
                        print(val)
                        #input()
                        if len(list(filter(lambda t: contains(val[0][0], t), sarr))):
                            return False
                else:
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if isinstance(val, list):val = val[0][0]
                        if val.find(s) < 0:
                            return False
            return True
        #print(func3(db))
        #input()
        return [x[1] for x in filter(func3, db)]


if __name__ == '__main__':
    db = Database()
    assert issubclass(Database, BaseDatabase)
    assert isinstance(db, BaseDatabase)
    res = db.query("餐馆", {"餐馆":{'评分':'4.5以上'}}, topk=3)
    from pprint import pprint
    pprint(res)
    print(len(res))
