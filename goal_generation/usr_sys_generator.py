import json, random, os, time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
from data.analysis import is_transactional, analysis_schema
import data.build_db as db
import pandas

domain_transition_matrix = [
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2]
]
'''
usr_matrix = [
    [0., 0.2, 0.3, 0.5, 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.], 
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.8, 0., 0.2, 0., 0., 0., 0.],
    [0., 0., 0., 0.5, 0., 0., 0., 0.5, 0., 0., 0.],
    [0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0.],
    [0., 0., 0.2, 0., 0., 0., 0.8, 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.8, 0.2],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
]
sys_matrix = [
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.7, 0., 0., 0.3],
    [0., 0., 0., 0.6, 0., 0.2, 0., 0.2, 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0.3, 0., 0., 0., 0.7, 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
]
usr_inner_matrix = np.array([
    [0.529, 0.   , 0.067, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.404],
    [0.   , 0.312, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.688],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.089, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.911],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.149, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.851],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.28 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.72 ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])
sys_inner_matrix = np.array([
    [0.278, 0.   , 0.059, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.663],
    [0.   , 0.198, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.802],
    [0.   , 0.   , 0.664, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.336],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.638, 0.   , 0.   , 0.   , 0.   , 0.   , 0.362],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])
'''
matrix_file = './matrix/matrix_weighted.npy'
with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]
        usr_matrix[1, 0] += usr_matrix[1, 4]; usr_matrix[1, 4]=0
        usr_matrix[-3, 2] += usr_matrix[-3, 5]; usr_matrix[-3, 5]=0
        usr_matrix[-4, 2] += usr_matrix[-4, 1]; usr_matrix[-4, 1]=0

        sys_matrix[3, -3] += sys_matrix[3, 1]; sys_matrix[3, 1]=0
        sys_matrix[3, 3] += sys_matrix[3, 4]; sys_matrix[3, 4]=0
        sys_matrix[2, 1] += sys_matrix[2, 4]; sys_matrix[2, 4]=0
        #sys_matrix[0, 1] += sys_matrix[0, 4]; sys_matrix[0, 4]=0

        

non_matrix_file = './matrix/non_matrix_weighted.npy'
with open(non_matrix_file, 'rb') as f:
        non_usr_matrix, non_sys_matrix, _, _ = [np.load(f) for _ in range(4)]

        non_usr_matrix[1, 0] += non_usr_matrix[1, 4]; non_usr_matrix[1, 4]=0        

        non_sys_matrix[-2, -3] += non_sys_matrix[-2, 2] + non_sys_matrix[-2, 5]; non_sys_matrix[-2, 2]=0; non_sys_matrix[-2, 5]=0
        non_sys_matrix[-1, 1] += non_sys_matrix[-1, 4]; non_sys_matrix[-1, 4]=0
        non_sys_matrix[2, 1] += non_sys_matrix[2, 2] + non_sys_matrix[2, 5]; non_sys_matrix[2, 2]=0; non_sys_matrix[2, 5]=0
        non_sys_matrix[5, -3] += non_sys_matrix[5, 1] + non_sys_matrix[5, 2] + non_sys_matrix[5, 5]; non_sys_matrix[5, 1]=0; non_sys_matrix[5, 2]=0; non_sys_matrix[5, 5]=0
        non_sys_matrix[4, -1] += non_sys_matrix[4, 1] + non_sys_matrix[4, 2] + non_sys_matrix[4, 4] + non_sys_matrix[4, 5] + non_sys_matrix[4, 7]; non_sys_matrix[4, 1]=0; non_sys_matrix[4, 2]=0; non_sys_matrix[4, 4]=0; non_sys_matrix[4, 5]=0; non_sys_matrix[4, 7]=0
        non_sys_matrix[3, -3] += non_sys_matrix[3, 1] + non_sys_matrix[3, 2] + non_sys_matrix[3, 5]; non_sys_matrix[3, 1]=0; non_sys_matrix[3, 2]=0; non_sys_matrix[3, 5]=0

usr_inner_matrix = np.array([
    [0.596, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.404],
    [0.   , 0.312, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.688],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.149, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.851],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.28 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.72 ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])

sys_inner_matrix = np.array([
    [0.278, 0.   , 0.059, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.663],
    [0.   , 0.198, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.802],
    [0.   , 0.   , 0.664, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.336],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])

usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]

def gen_init_state(value_list, intent_i):
    usr_idx, sys_idx = 3, 3
    usr_idx = np.random.choice([i for i in range(len(non_usr_matrix[sys_idx]))], 1, p=non_usr_matrix[sys_idx])[0]
    usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
    old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)

    usr_mat, sys_mat = is_trans(value_list, intent_i)
    sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]    
    sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
    old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
    return (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat)

def is_trans(value_list, intent_i):
    if is_transactional(value_list, intent_i["name"]):
        usr_mat = usr_matrix
        sys_mat = sys_matrix                    
    else: 
        usr_mat = non_usr_matrix
        sys_mat = non_sys_matrix
    return usr_mat, sys_mat

def get_usr_actions(acts, slots, domain, intent):
    service_dict = {"Mail_1": 'data/csv/mail_entityies.csv', "Calendar_1": 'data/csv/events.csv', "Messaging_1": 'data/csv/message_entityies.csv'}
    df = pandas.read_csv(service_dict[domain])
    data = df.sample(replace=True, random_state=int(time.time()))
    slot_values = data.to_dict('records')[0]
    slots = set(list(slots))
    actions = []
    #print(acts, slots)
    #input()
    for act in list(set(acts)):
        tmp_acts = []       
        if act == "none":continue         
        elif act == "INFORM":            
            for slot in slots:
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(data[slot].values[0])
                act_dict["values"].append(data[slot].values[0])      
                tmp_acts.append(act_dict)                  
        elif act == "REQUEST":
            for slot in slots:
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                tmp_acts.append(act_dict)  
        elif act == "INFORM_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
            act_dict["canonical_values"].append(intent["name"])
            act_dict["values"].append(intent["name"])
            tmp_acts.append(act_dict)
        else: 
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
        actions += tmp_acts
        #print(act_dict, slots)
        #if act!='none': print(act, intent["name"])
    #print('usr', actions)
    return actions, slot_values

def get_sys_actions(acts, slot_values, domain, intent):
    service_dict = {"Mail_1": 'data/csv/mail_entityies.csv', "Calendar_1": 'data/csv/events.csv', "Messaging_1": 'data/csv/message_entityies.csv'}
    df = pandas.read_csv(service_dict[domain])
    #data = df.sample(replace=True, random_state=1)
    results = db.find_result(domain, slot_values)
    actions = []
    #print(acts, slot_values)
    #input()
    for act in list(set(acts)):  
        tmp_acts = []   
        if act == "none":continue          
        elif act == "INFORM":            
            for slot, value in list(slot_values.items()):
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value)
                act_dict["values"].append(value)      
                tmp_acts.append(act_dict)                  
        elif act == "REQUEST":
            for slot, value in slot_values.items():
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                tmp_acts.append(act_dict)  
        elif act == "OFFER":
            for slot, value in slot_values.items():
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value)
                act_dict["values"].append(value)      
                if not act_dict in tmp_acts: tmp_acts.append(act_dict)
        elif act == "CONFIRM":  
            for slot, value in set(list(slot_values.items())):
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value)
                act_dict["values"].append(value)      
                tmp_acts.append(act_dict)
        elif act == "INFORM_COUNT":  
            act_dict = {"act": act, "canonical_values": [len(results)], "slot": "count", "values": [len(results)]}
            tmp_acts.append(act_dict)
        elif act == "OFFER_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
            act_dict["canonical_values"].append(intent["name"])
            act_dict["values"].append(intent["name"])
            tmp_acts.append(act_dict)
        else: 
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
        actions += tmp_acts
        #if act!='none': print(act, intent["name"])
    #print('sys', actions)
    return actions

def print_actions(u_acts, s_acts, intent_i):
    print(u_acts[0], intent_i["name"])
    if u_acts[1]!='none': print(u_acts[1])
    print(s_acts[0])
    if s_acts[1]!='none': print(s_acts[1])

def usr_act2robot_utt(usr_actions):
    template, n = "", 0
    vocab = {'findcontact': '找聯絡人', 'findmessage': '找訊息', 'sendmessage': '傳送訊息', 'message': '訊息', 'contact_name': '聯絡人', 'app_name': '應用程式',\
        'event_date': '活動日期', 'event_time': '活動時間', 'event_location':'活動地點', 'event_name':'活動名稱', 'event_content':'活動內容',\
        'participant':'參加者', 'available_start_time':'開始時間', 'available_end_time':'結束時間',\
        'getevents':'知道相關的活動', 'lookupevents':'找活動', 'getavailabletime':'有空的時間', 'addevent':'添加一個活動',\
        'recipient':'收件者', 'sender':'寄件者', 'subject':'主旨', 'content':'內容', 'copy_recipient':'副本收件者', \
        'sendmail':'寄一封信', 'findmail':'找一封信', 'REQUEST':'我想知道', 'THANK_YOU':'', 'INFORM_INTENT':'我想要', 'INFORM':'資訊如下:\n',\
        'AFFIRM':'', 'SELECT':'', 'NEGATE':'', 'REQUEST_ALTS':'', 'GOODBYE':'再見', 'NEGATE_INTENT':'', 'AFFIRM_INTENT':'好的，麻煩你了'}
    #print(usr_actions)
    for u_act in usr_actions:
        act, _, slot, value = list(u_act.values())
        #print(act)
        if n==0: template += vocab[act]
        if act == "INFORM_INTENT": template += f"{vocab[value[0].lower()]}。"
        elif act == "INFORM": template += f"{vocab[slot.lower()]}:{value[0]}，"
        elif act == "REQUEST": template += f"{vocab[slot.lower()]}、"
        elif act == "THANK_YOU": template += f"謝謝!"
        elif act == "AFFIRM": template += f"是的，沒有問題"
        elif act == "SELECT": template += f"好的"
        elif act == "NEGATE": template += f"不"
        elif act == "REQUEST_ALTS": template += f"還有其他的嗎"
        elif act == "GOODBYE": pass
        elif act == "NEGATE_INTENT": template += f"不用，沒關係"        
        elif act == "AFFIRM_INTENT": pass
        n += 1
    print("使用者:", template, '\n')
    return template

def sys_act2robot_utt(sys_actions):
    template, n = "", 0
    vocab = {'findcontact': '找聯絡人', 'findmessage': '找訊息', 'sendmessage': '傳送訊息', 'message': '訊息', 'contact_name': '聯絡人', 'app_name': '應用程式',\
        'event_date': '活動日期', 'event_time': '活動時間', 'event_location':'活動地點', 'event_name':'活動名稱', 'event_content':'活動內容',\
        'participant':'參加者', 'available_start_time':'開始時間', 'available_end_time':'結束時間',\
        'getevents':'知道相關的活動', 'lookupevents':'找活動', 'getavailabletime':'有空的時間', 'addevent':'添加一個活動',\
        'recipient':'收件者', 'sender':'寄件者', 'subject':'主旨', 'content':'內容', 'copy_recipient':'副本收件者', \
        'sendmail':'寄一封信', 'findmail':'找一封信', 'REQUEST':'請問', 'INFORM_COUNT':'找到以下', 'INFORM':'資訊如下:\n',\
        'OFFER':'這是', 'CONFIRM':'請確認以下資訊是否正確？\n', 'NOTIFY_SUCCESS':'已達成', 'REQ_MORE':'', 'GOODBYE':'再會', 'OFFER_INTENT':'您會想要', \
        'NOTIFY_FAILURE': '不好意思，無法達到您的'}
    #print(sys_actions)
    for s_act in sys_actions:        
        act, _, slot, value = list(s_act.values())
        #print(act)
        if n==0: template += vocab[act]
        if act == "INFORM": template += f"{vocab[slot.lower()]}:{value[0]}，"
        elif act == "REQUEST": template += f"{vocab[slot.lower()]}、"
        elif act == "OFFER": template += f"{vocab[slot.lower()]}、"
        elif act == "CONFIRM": template += f"{vocab[slot.lower()]}:{value[0]}，"
        elif act == "INFORM_COUNT": template += f"{str(value[0])}的結果"
        elif act == "NOTIFY_SUCCESS": template += f"需求"
        elif act == "REQ_MORE": template += f"需要其他服務嗎？"
        elif act == "GOODBYE": pass
        elif act == "OFFER_INTENT": template += f"{vocab[value[0].lower()]}嗎？"        
        elif act == "NOTIFY_FAILURE": template += f"需求"  
        n += 1  
    print("助理:",template, '\n')
    return template

def generate_goal(file_name):
    domain_index = 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8").read())
    value_list = analysis_schema(file_name)
    domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
    state_d = schemas[domain_index]                            
    intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
    slot_i = list(intent_i["required_slots"])
    #print(state_d["service_name"], intent_i["name"], list(intent_i["optional_slots"].keys()))
    if len(slot_i)==0 and len(list(intent_i["optional_slots"].keys()))!=0: slot_i = \
        random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))
    
    #initial state
    (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)
    usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
    sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
    while True:     
        #print_actions((usr_act, usr_act_inner), (sys_act, sys_act_inner), intent_i)
        usr_actions, slot_values = get_usr_actions([usr_act, usr_act_inner], slot_i, state_d['service_name'], intent_i)
        sys_actions = get_sys_actions([sys_act, sys_act_inner], slot_values, state_d['service_name'], intent_i)
        #print(usr_actions, slot_values)
        usr_act2robot_utt(usr_actions)
        #print(sys_actions)
        sys_act2robot_utt(sys_actions)
        usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
        usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
        old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)
        
        sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
        sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
        old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
        usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
        sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
        
        if usr_act == "INFORM_INTENT":
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                            
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            slot_i = list(intent_i["required_slots"])
            if len(slot_i)==0 and len(list(intent_i["optional_slots"].keys()))!=0: slot_i = \
                random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))            
            (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)

        if sys_act == "OFFER_INTENT":   
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                            
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            slot_i = list(intent_i["required_slots"])
            if len(slot_i)==0 and len(list(intent_i["optional_slots"].keys()))!=0: slot_i = \
                random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))
            (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)
            
        if sys_act == "GOODBYE" or sys_act_inner == "GOODBYE":
            usr_actions, slot_values = get_usr_actions((usr_act, usr_act_inner), slot_i, state_d['service_name'], intent_i)
            sys_actions = get_sys_actions((sys_act, sys_act_inner), slot_values, state_d['service_name'], intent_i)
            usr_act2robot_utt(usr_actions)
            sys_act2robot_utt(sys_actions)
            break    

def draw_matrix(matrix, select_acts=[], output_acts=[], labels=[], img_name='default'):
    
    fig, ax = plt.subplots()
    plt.title(img_name)
    im = ax.imshow(matrix, cmap='OrRd', aspect='auto')
    # Major ticks
    ax.set_xticks(np.arange(0, len(matrix[0]), 1))
    ax.set_yticks(np.arange(0, len(matrix), 1))    

    # Labels for major ticks
    ax.set_xticklabels(output_acts, rotation=90, color="red" if labels[0]!='Client' else 'blue')
    ax.set_yticklabels(select_acts, rotation=0, color="red" if labels[1]!='Client' else 'blue')    
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, color="red" if labels[0]!='Client' else 'blue')
    ax.tick_params(axis="y", color="red" if labels[1]!='Client' else 'blue')
    ax.set_xlabel('decide action'+f"({labels[0]})", color="red" if labels[0]!='Client' else 'blue')
    ax.set_ylabel('select action'+f"({labels[1]})", color="red" if labels[1]!='Client' else 'blue')    

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(matrix[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matrix), 1), minor=True)  

    fig.colorbar(im)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            c = matrix[i, j]
            if c > 0:
                ax.text(j, i, str(c), va='center', ha='center')
    plt.grid(which='minor', color='k', linewidth=1)
    fig.tight_layout()
    plt.savefig(os.path.join('../transistion matrix', img_name))
    plt.show()

if __name__ == "__main__":
    generate_goal("./schema/messagewoz_schema.json")
    #print(usr_matrix)
    #print(sys_matrix)

    '''
    usr_matrix = np.around(usr_matrix, decimals=2)    
    draw_matrix(usr_matrix, sys_acts[:-1], usr_acts[:-1], ['Client', 'Assistant'], 'Decide Client (transactional)')

    sys_matrix = np.around(sys_matrix, decimals=2)
    draw_matrix(sys_matrix, usr_acts[:-1], sys_acts[:-1], ['Assistant', 'Client'], 'Decide Assistant (transactional)')

    non_usr_matrix = np.around(non_usr_matrix, decimals=2)    
    draw_matrix(non_usr_matrix, sys_acts[:-1], usr_acts[:-1], ['Client', 'Assistant'], 'Decide Client (non-transactional)')

    non_sys_matrix = np.around(non_sys_matrix, decimals=2)
    draw_matrix(non_sys_matrix, usr_acts[:-1], sys_acts[:-1], ['Assistant', 'Client'], 'Decide Assistant (non-transactional)')

    usr_inner_matrix = np.around(usr_inner_matrix, decimals=2)
    draw_matrix(usr_inner_matrix, usr_acts, usr_acts, ['Client', 'Client'], 'Continue Decide Client')

    sys_inner_matrix = np.around(sys_inner_matrix, decimals=2)
    draw_matrix(sys_inner_matrix, sys_acts, sys_acts, ['Assistant', 'Assistant'], 'Continue Decide Assistant')
    '''
    