"""
from convlab2.util.crosswoz.dbquery import Database
db = Database()
db.query(state['belief_state'], cur_domain)
"""
import json
import os
import re
from pprint import pprint
from collections import Counter


def contains(arr, s):
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))


class Database(object):
    """docstring for Database"""

    def __init__(self):
        super(Database, self).__init__()

        self.data = {}
        db_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../data/unified_datasets/messagewoz/1_fold_test/database'))
        with open(os.path.join(db_dir, 'gcalendar_db.json'), 'r', encoding='utf-8') as f:
            self.data['Calendar'] = json.load(f)
        with open(os.path.join(db_dir, 'gmail_db.json'), 'r', encoding='utf-8') as f:
            self.data['Gmail'] = json.load(f)

        self.schema = {
            'Calendar': {
              '活動名稱': { 'params': None },
              '活動時間': { 'type': 'between', 'params': [None, None] },
              '參加者': { 'type': 'multiple_in', 'params': None },
              '是否全天': { 'type': 'choose', 'params': None },
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

    def query(self, belief_state, cur_domain):
        """
        query database using belief state, return list of entities, same format as database
        :param belief_state: state['belief_state']
        :param cur_domain: maintain by DST, current query domain
        :return: list of entities
        """
        if not cur_domain or cur_domain == 'Message':
            return []
        cur_query_form = {}
        for slot, value in belief_state[cur_domain].items():
            if not value: continue
            cur_query_form[slot] = value
        cur_res = self.query_schema(field=cur_domain, args=cur_query_form)
        res = cur_res

        return res

    def query_schema(self, field, args):
        if not field in self.schema:
            raise Exception('Unknown field %s' % field)
        if not isinstance(args, dict):
            raise Exception('`args` must be dict')
        db = self.data.get(field)
        plan = self.schema[field]
        for key, value in args.items():
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
            details = item[1]
            for key, val in args.items():
                val = details.get(key)
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
                    if not s is None:
                        if absence:
                            return False
                        sarr = list(filter(lambda t: bool(t), s.split(' ')))
                        if len(list(filter(lambda t: contains(val, t), sarr))):
                            return False
                else:
                    s = options['params']
                    if not s is None:
                        if absence:
                            return False
                        if val.find(s) < 0:
                            return False
            return True
        return list(filter(func3, db))


if __name__ == '__main__':
    db = Database()
    #state = default_state()
    dishes = {}
    for n,v in db.query(state['belief_state'], '餐馆'):
        for dish in v['推荐菜']:
            dishes.setdefault(dish, 0)
            dishes[dish]+=1
    pprint(Counter(dishes))
