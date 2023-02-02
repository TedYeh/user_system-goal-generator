import json, random, os, time
import re
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
from data.analysis import is_transactional, analysis_schema
import data.build_db as db
import pandas as pd
#from helper import get_query_text

domain_transition_matrix = [
        [0.05, 0.5 , 0.45],
        [0.45, 0.05, 0.5 ],
        [0.5 , 0.45, 0.05]
]

MAX_GOAL = 3

tagger_path = './tagger'
if not os.path.isdir(tagger_path):
    from ckiptagger import data_utils
    data_utils.download_data_gdown("./")

'''
matrix_file = './matrix/matrix_weighted.npy'
with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]

non_matrix_file = './matrix/non_matrix_weighted.npy'
with open(non_matrix_file, 'rb') as f:
        non_usr_matrix, non_sys_matrix, _, _ = [np.load(f) for _ in range(4)]
'''

usr_matrix = np.array([
    [0.   , 0.126, 0.045, 0.486, 0.   , 0.138, 0.   , 0.   , 0.205, 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.323, 0.336, 0.   , 0.136, 0.   , 0.   , 0.205, 0.   , 0.   ],
    [0.   , 0.   , 0.15 , 0.   , 0.   , 0.   , 0.85 , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.7  , 0.3  ],
    [0.   , 0.   , 0.   , 0.   , 0.898, 0.   , 0.056, 0.   , 0.046, 0.   , 0.   ]
])

sys_matrix = np.array([
    [0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.508, 0.   , 0.   , 0.   , 0.492, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.83 , 0.17 , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.344, 0.   , 0.   , 0.   , 0.656, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.977, 0.   , 0.   , 0.023, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]
])

non_usr_matrix = np.array([
    [0.   , 0.126, 0.166, 0.038, 0.   , 0.401, 0.   , 0.205, 0.064, 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.277, 0.   , 0.007, 0.   , 0.463, 0.   , 0.207, 0.046, 0.   , 0.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.322, 0.   , 0.006, 0.   , 0.432, 0.   , 0.192, 0.048, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.15 , 0.   , 0.   , 0.   , 0.85 , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.617, 0.383],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.908, 0.   , 0.092, 0.   , 0.   ]
])

non_sys_matrix = np.array([
    [0.   , 0.   , 0.547, 0.   , 0.   , 0.453, 0.   , 0.   , 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.615, 0.   , 0.   , 0.   , 0.385, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.77 , 0.   , 0.   , 0.23 ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.367, 0.633, 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
    [0.   , 0.   , 0.763, 0.   , 0.   , 0.217, 0.   , 0.02 , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]
])

usr_inner_matrix = np.array([
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.332, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.668],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.05 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.95 ],
    [0.   , 0.   , 0.   , 0.15 , 0.   , 0.   , 0.   , 0.   , 0.85 , 0.   , 0.   , 0.   ],
    [0.35 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.65 ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.28 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.72 ],
    [0.32 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.68 ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])

sys_inner_matrix = np.array([
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.198, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.802],
    [0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ]
])

usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]

def gen_init_state(value_list, intent_i, idxs=[]):
    usr_mat, sys_mat = is_trans(value_list, intent_i)
    usr_idx, sys_idx = [3, 3] if not idxs else tuple(idxs) 
    usr_idx = np.random.choice([i for i in range(len(usr_mat[sys_idx]))], 1, p=usr_mat[sys_idx])[0]
    usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
    old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)
    
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

def change_slot(slot, slot_value):
    import calendar    
    slot_templates = json.loads(open('./template/slot_change.json', 'r', encoding='utf-8').read())
    if slot == "event_date":
        weekday_dict, date_format ={0:"一", 1:"二", 2:"三", 3:"四", 4:"五", 5:"六", 6:"日"}, {}
        year, month, day = [int(date_value) for date_value in slot_value.split('/')]
        date_format["year"] = year; date_format["month"] = month; date_format["day"] = day
        date_format["weekday"] = weekday_dict[calendar.weekday(year, month, day)]
        slot_value = random.choice(slot_templates[slot])
        for date_slot, date_value in date_format.items():
            slot_value = slot_value.replace(f'[{date_slot}]', str(date_value))        
        return slot_value
    elif slot == "event_time":
        time_format = {}
        hour, minute = [int(t) for t in slot_value.split(':')]
        if hour < 12: tw_clk = "上午" 
        else: 
            hour = hour - 12 if hour != 12 else 12 
            tw_clk = "下午"
        time_format["hour"] = hour; time_format["minute"] = minute; time_format["tw_clk"] = tw_clk
        slot_value = random.choice(slot_templates[slot])
        for time_slot, time_value in time_format.items():
            slot_value = slot_value.replace(f'[{time_slot}]', str(time_value))
        return slot_value
    else: return slot_value    

def get_usr_actions(acts, slots_, intent, data, state={}):
    slot_values = {}
    requested_slots = []
    slots = sorted(list(set(slots_[0] + slots_[1])), key=lambda x:find_slot_weight(state['slots'], x), reverse=True)
    actions, annotations = [], []    
    for act in list(set(acts)):
        tmp_acts = []       
        if act == "none":continue       
        elif act == "INFORM":              
            ann_str = ''                   
            for slot in slots:  
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []} 
                values = change_slot(slot, data[slot].values[0])
                act_dict["canonical_values"].append(data[slot].values[0])
                act_dict["values"].append(values) 
                tmp_acts.append(act_dict)
                slot_values[slot] = [values]
                ann_str += f"{act}({slot}=〔{values}〕) "
            annotations.append(ann_str)                 
        elif act == "REQUEST":
            #print(slots, slots_[2])
            for slot in slots: 
                if slot in slots_[2]: slots_[2].remove(slot)
            if slots_[2]: slots = list(set(random.choices(list(slots_[2]), k=random.randint(1, len(list(slots_[2]))))))
            ann_str = ''
            for slot in slots:
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                slot_values[slot] = [data[slot].values[0]]
                tmp_acts.append(act_dict)  
                ann_str += f"{act}({slot}) "
                if slot in slots_[2]: slots_[2].remove(slot)
            annotations.append(ann_str)
            requested_slots = list(slots)
        elif act == "INFORM_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
            act_dict["canonical_values"].append(intent["name"])
            act_dict["values"].append(intent["name"])
            tmp_acts.append(act_dict)
            annotations.append(f"{act}(" + f"{intent['name']})")
            for slot in slots:  
                slot_values[slot] = [data[slot].values[0]]
        elif act=="AFFIRM_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}()")
            for slot in slots:  
                slot_values[slot] = [data[slot].values[0]]
        elif act=="AFFIRM":
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}()")
            for slot in slots:  
                slot_values[slot] = [data[slot].values[0]]
        else: 
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}()")
        if act == "REQUEST_ALTS":
            for slot in slots: slot_values[slot] = [data[slot].values[0]]
        elif act == "NEGATE":
            for slot in slots: slot_values[slot] = [data[slot].values[0]]
        actions += tmp_acts
    return actions, slot_values, slots, 'U: ' + ' '.join(annotations), requested_slots

def get_sys_actions(acts, slots, slot_values, state, intent, df):
    domain = state['service_name']
    service_call = None
    service_results = []  
    columns = list(df.columns)              
    actions, annotations = [], []    
    for act in list(set(acts)):  
        tmp_acts = []  
        if act == "none":continue          
        elif act == "INFORM": 
            ann_str = ''   
            req_results = db.find_result_slot(domain, slot_values, slots[0])
            for k, v in zip(list(slots[0]), req_results[-1]):
                act_dict = {"act": act, "canonical_values": [], "slot": k, "values": []}
                act_dict["canonical_values"].append(v)
                act_dict["values"].append(v)      
                tmp_acts.append(act_dict) 
                ann_str += f"{act}({k}=〔{v}〕) "
            annotations.append(ann_str)        
        elif act == "REQUEST":
            ann_str = ''
            for slot in sorted(slots[0], key=lambda x:find_slot_weight(state['slots'], x), reverse=True):
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                tmp_acts.append(act_dict) 
                ann_str += f"{act}({slot}) "
            annotations.append(ann_str) 
        elif act == "OFFER":
            ann_str = ''
            #print('sys', slot_values, act)
            service_call = {"method":intent["name"], "parameters":{}}
            results = db.find_result(domain, slot_values) #, slots[0]
            if slot_values:                 
                tmp_dict = {}
                for res in results:
                    for k, v in zip(list(columns), res):
                        tmp_dict[k] = v
                service_results.append(deepcopy(tmp_dict))
            for slot, value in slot_values.items():
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value[0])
                act_dict["values"].append(value[0])     
                service_call["parameters"][slot] = value
                if not act_dict in tmp_acts: tmp_acts.append(act_dict)
                ann_str += f"{act}({slot}=〔{value[0]}〕) "
            annotations.append(ann_str)
        elif act == "CONFIRM": 
            ann_str = ''
            for slot, value in list(slot_values.items()):
                #if value == "無":continue
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value[0])
                act_dict["values"].append(value[0])      
                tmp_acts.append(act_dict)
                ann_str += f"{act}({slot}=〔{value[0]}〕) "
            annotations.append(ann_str)
        elif act == "INFORM_COUNT":  
            service_call = {"method":intent["name"], "parameters":{}}  
            results = db.find_result(domain, slot_values) 
            for slot, value in slot_values.items(): service_call["parameters"][slot] = value[0]
            act_dict = {"act": act, "canonical_values": [str(len(results))], "slot": "count", "values": [str(len(results))]}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}(" + f"{len(results)})")
        elif act == "OFFER_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
            act_dict["canonical_values"].append(intent["name"])
            act_dict["values"].append(intent["name"])
            tmp_acts.append(act_dict)
            annotations.append(f"{act}(" + f"{intent['name']})")
        else: 
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}()")
        if act == "NOTIFY_SUCCESS" or act == "NOTIFY_FAILURE":
            service_call = {"method":intent["name"], "parameters":{}}
            for slot, value in list(slot_values.items()): service_call["parameters"][slot] = value[0]
        actions += tmp_acts
        #if act!='none': print(act, intent["name"])
    #print('sys', actions)
    return actions, 'S: ' + ' '.join(annotations), service_call, service_results

def print_actions(u_acts, s_acts, intent_i):
    print(u_acts[0], intent_i["name"])
    if u_acts[1]!='none': print(u_acts[1])
    print(s_acts[0])
    if s_acts[1]!='none': print(s_acts[1])

def find_slot(uttrance, acts):
    slots = []    
    for a in acts:        
        act, _, slot, value = list(a.values())
        if act == "INFORM": 
            if value[0]:
                for match in re.finditer(re.escape(value[0]), uttrance):
                    slots.append({"exclusive_end": match.end(), "slot": slot, "start": match.start()})
    return slots

def find_slot_weight(slot_dicts, slot):
    for slot_dict in slot_dicts:
        if slot_dict['name'] == slot:
            return slot_dict['weight']
    return 0

def get_uttr_list(slot_act_dict, state, template_list):
    uttr_templates = []
    for act in slot_act_dict.keys():
        slots = sorted(list(slot_act_dict[act].keys()), key=lambda x:find_slot_weight(state['slots'], x), reverse=True)
        for temp in sorted(list(template_list[act].keys()), key=len, reverse=False):
            if all(elem in slots for elem in temp.split('+')) and slots and len(slots)>=len(temp.split('+')):
                uttr = random.choice(template_list[act][temp])                 
                for s in temp.split('+'):
                    if act=="INFORM" or act=="CONFIRM" or act=="INFORM_COUNT" or act=="OFFER":uttr = uttr.replace(f'[{s}]', str(slot_act_dict[act][s]))
                    else:uttr = random.choice(template_list[act][temp])
                    slots.remove(s)  
                uttr_templates.append(uttr)  
    return uttr_templates 

def usr_act2robot_utt(usr_actions, state, uttr_template):
    if list(usr_actions[-1].values())[0] in ["INFORM_INTENT", "NEGATE", "SELECT"]:
        usr_actions[0], usr_actions[-1] = usr_actions[-1], usr_actions[0]
    #print('usr', usr_actions)
    domain = state['service_name']
    slots, slot_act_dict = [], {}
    templates = json.loads(open(uttr_template, 'r', encoding='utf-8').read())
    for u_act in usr_actions:        
        act, _, slot, value = list(u_act.values()) 
        if not act in slot_act_dict: slot_act_dict[act]={}       
        if act == "INFORM_INTENT":
            slot_act_dict[act][value[0]] = value[0]
        else:
            if '無' in value: slot += '*無'  
            if slot == '': slot='none' 
            slot_act_dict[act][slot] = value[0] if value else 'none'   
    uttr_templates = get_uttr_list(slot_act_dict, state, templates[domain])
    template = '，'.join(uttr_templates)
    slots_pos = find_slot(template, usr_actions)
    print('使用者：', template)
    return template, slots_pos

def sys_act2robot_utt(sys_actions, state, uttr_template, active_intent=''):
    #print('sys', sys_actions)
    domain = state['service_name']
    slots, slot_act_dict = [], {}
    templates = json.loads(open(uttr_template, 'r', encoding='utf-8').read())
    for s_act in sys_actions:        
        act, _, slot, value = list(s_act.values())
        if not act in slot_act_dict: slot_act_dict[act]={}  
        if act=="OFFER_INTENT":
            slot_act_dict[act][value[0]] = value[0]
        elif act in ["NOTIFY_SUCCESS", "NOTIFY_FAILURE"]:
            slot_act_dict[act][active_intent] = active_intent
        else:
            if '無' in value: slot += '*無'
            if slot == '': slot='none'
            slot_act_dict[act][slot] = value[0] if value else 'none' 
    uttr_templates = get_uttr_list(slot_act_dict, state, templates[domain])
    template = '，'.join(uttr_templates)
    slots_pos = find_slot(template, sys_actions)
    print('助理：', template)
    return template, slots_pos

def generate_goal(file_name, file_idx):
    service_dict = {"Mail_1": 'data/csv/mail_entityies.csv', "Calendar_1": 'data/csv/events.csv', "Messaging_1": 'data/csv/message_entityies_line.csv'}
    dialogues = {"dialogue_id": file_idx, "services": [], "turns": []}
    domain_index, dialogs, goal_count = 0, {'Annotation(Actions)':[], 'Template utterances':[]}, 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8-sig").read())
    usr_dialog_state_his, sys_dialog_state_his = [], []
    value_list = analysis_schema(file_name)
    domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0]     
    state_d = schemas[domain_index]     
    dialogues["services"].append(state_d['service_name'])                       
    intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
    req_slot_i = list(intent_i["required_slots"])
    opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
        if len(list(intent_i["optional_slots"].keys()))!=0 else []
    result_slots = list(intent_i["result_slots"])
    df = pd.read_csv(service_dict[state_d['service_name']], encoding='utf-8-sig')
    if state_d['service_name']=="Messaging_1":
        for slot in opt_slot_i: data = df.loc[df[slot] != '無'].sample(replace=False, random_state=round(time.time()))
    else:
        data = df.sample(replace=False, random_state=round(time.time()))
    
    #initial state
    (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)
    usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
    sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
    while True:   
        usr_turn, sys_turn = {"frames":[], "speaker": "USER", "utterance":""}, {"frames":[], "speaker": "SYSTEM", "utterance":""}  
        usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
        #print_actions((usr_act, usr_act_inner), (sys_act, sys_act_inner), intent_i)
        #print([usr_act, usr_act_inner])
        if ("REQUEST" in [sys_act, sys_act_inner]):
            usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data, state_d)
            for u_action in usr_actions: usr_dialog_state_his.append(list(u_action.keys())[0:2])
            u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, state_d, './template/usr_template.json')
            usr_frames['state']['active_intent'] = deepcopy(intent_i['name'])
            usr_frames['state']['requested_slots'] = deepcopy(requested_slots)
            usr_frames['state']['slot_values'] = {}
            usr_frames["actions"] = deepcopy(list(usr_actions))
            usr_turn["utterance"] = deepcopy(u_uttr)
            usr_frames['slots'] = deepcopy(u_slots_pos)
            usr_turn["frames"] = [deepcopy(usr_frames)]
            dialogues["turns"].append(deepcopy(usr_turn))
            dialogs['Annotation(Actions)'].append(usr_annotations); dialogs['Template utterances'].append(u_uttr)
            tmp_slot_values = deepcopy(slot_values)
            usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
            usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
            usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]  
            slots_ = list(set(req_slot_i + opt_slot_i))
            while len(slots_)!=0:
                usr_turn, sys_turn = {"frames":[], "speaker": "USER", "utterance":""}, {"frames":[], "speaker": "SYSTEM", "utterance":""}  
                usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
                select_slot = list(set(random.choices(slots_, k=random.randint(1, 3)))) #最多詢問三項資訊
                sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [select_slot, result_slots], slot_values, state_d, intent_i, df)
                for s_action in sys_actions: sys_dialog_state_his.append(list(s_action.keys())[0:2])
                usr_actions, slot_values, _, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [select_slot, select_slot, result_slots], intent_i, data, state_d)
                for u_action in usr_actions: usr_dialog_state_his.append(list(u_action.keys())[0:2])
                s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, state_d, './template/sys_template.json', deepcopy(intent_i['name']))
                u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, state_d, './template/usr_template.json')               

                sys_frames["actions"] = deepcopy(list(sys_actions))
                if service_call: sys_frames['service_call'] = service_call
                if service_results: sys_frames['service_results'] = service_results
                sys_frames['slots'] = deepcopy(s_slots_pos)
                sys_turn["frames"] = [deepcopy(sys_frames)]
                sys_turn["utterance"] = deepcopy(s_uttr)
                dialogues["turns"].append(deepcopy(sys_turn))
                dialogs['Annotation(Actions)'].append(sys_annotations); dialogs['Template utterances'].append(s_uttr)

                usr_frames['state']['active_intent'] = deepcopy(intent_i['name'])
                usr_frames['state']['requested_slots'] = deepcopy(requested_slots)
                usr_frames['state']['slot_values'] = deepcopy(slot_values)
                usr_frames["actions"] = deepcopy(list(usr_actions))
                usr_frames['slots'] = deepcopy(u_slots_pos)
                usr_turn["frames"] = [deepcopy(usr_frames)]
                usr_turn["utterance"] = deepcopy(u_uttr)
                dialogues["turns"].append(deepcopy(usr_turn))
                dialogs['Annotation(Actions)'].append(usr_annotations); dialogs['Template utterances'].append(u_uttr)
                for s in select_slot: slots_.remove(s)

            sys_turn = {"frames":[], "speaker": "SYSTEM", "utterance":""}  
            sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': []}
                
            old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)        
            sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
            sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
            old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)            
            sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner] 
            sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [list(set(req_slot_i+opt_slot_i)), result_slots], tmp_slot_values, state_d, intent_i, df)
            s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, state_d, './template/sys_template.json', deepcopy(intent_i['name']))
            sys_frames["actions"] = deepcopy(list(sys_actions))
            if service_call: sys_frames['service_call'] = service_call
            if service_results: sys_frames['service_results'] = service_results
            sys_frames['slots'] = deepcopy(s_slots_pos)
            sys_turn["frames"] = [deepcopy(sys_frames)]
            sys_turn["utterance"] = deepcopy(s_uttr)
            dialogues["turns"].append(deepcopy(sys_turn))
            dialogs['Annotation(Actions)'].append(sys_annotations); dialogs['Template utterances'].append(s_uttr)

            usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
            usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
            old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx) 
            usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]

            usr_turn, sys_turn = {"frames":[], "speaker": "USER", "utterance":""}, {"frames":[], "speaker": "SYSTEM", "utterance":""}  
            usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
               
            sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
            sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0] 
            old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)                                   
            sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner] 
            
        usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data, state_d)
        for u_action in usr_actions: 
            if list(u_action.keys())[0:2] in [usr_dialog_state_his]:
                usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
                usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
                old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx) 
                usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
                usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data, state_d)
                for u_action_ in usr_actions: usr_dialog_state_his.append(list(u_action_.keys())[0:2])
                break
        for u_action in usr_actions: usr_dialog_state_his.append(list(u_action.keys())[0:2])

        sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [slots, result_slots], slot_values, state_d, intent_i, df)
        for s_action in sys_actions: 
            if list(s_action.keys())[0:2] in [sys_dialog_state_his]:
                sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
                sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0] 
                old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)                                   
                sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
                sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [slots, result_slots], slot_values, state_d, intent_i, df)
                for s_action_ in sys_actions: sys_dialog_state_his.append(list(s_action_.keys())[0:2])
                break
        for s_action in sys_actions: sys_dialog_state_his.append(list(s_action.keys())[0:2])                

        #------------------------------Record data------------------------------
        if  "NEGATE" in [usr_act, usr_act_inner] or "NEGATE_INTENT" in [usr_act, usr_act_inner]:
            usr_frames['state']['active_intent'] = "NONE"
        else:
            usr_frames['state']['active_intent'] = deepcopy(intent_i['name'])
        usr_frames['state']['requested_slots'] = deepcopy(requested_slots)
        usr_frames['state']['slot_values'] = deepcopy(slot_values)
        usr_frames["actions"] = deepcopy(list(usr_actions))
        sys_frames["actions"] = deepcopy(list(sys_actions))
        if service_call: sys_frames['service_call'] = service_call
        if service_results: sys_frames['service_results'] = service_results
        u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, state_d, './template/usr_template.json')
        s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, state_d, './template/sys_template.json', deepcopy(intent_i['name']))
        usr_turn["utterance"] = deepcopy(u_uttr)
        sys_turn["utterance"] = deepcopy(s_uttr)

        usr_frames['slots'] = deepcopy(u_slots_pos)
        sys_frames['slots'] = deepcopy(s_slots_pos)

        usr_turn["frames"] = [deepcopy(usr_frames)]
        sys_turn["frames"] = [deepcopy(sys_frames)]   
            
        dialogues["turns"].append(deepcopy(usr_turn))
        dialogues["turns"].append(deepcopy(sys_turn))
        

        dialogs['Annotation(Actions)'].append(usr_annotations); dialogs['Template utterances'].append(u_uttr)
        dialogs['Annotation(Actions)'].append(sys_annotations); dialogs['Template utterances'].append(s_uttr)
        usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
        if goal_count >=3:
            if usr_idx == 2: usr_idx = 6
            elif usr_idx == 10: usr_idx = 9
        usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
        old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)
         
        sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
        sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
        old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
        usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
        sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
            
        if "REQUEST_ALTS" in [usr_act, usr_act_inner]:
            df = pd.read_csv(service_dict[state_d['service_name']], encoding='utf-8-sig')
            data = df.sample(replace=False, random_state=round(time.time()))

        if "INFORM_INTENT" in [usr_act, usr_act_inner]:
            goal_count += 1
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0]             
            state_d = schemas[domain_index] 
            
            if not state_d['service_name'] in dialogues["services"]: dialogues["services"].append(state_d['service_name'])                
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            req_slot_i = list(intent_i["required_slots"])
            opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
                if len(list(intent_i["optional_slots"].keys()))!=0 else []  
            result_slots = list(intent_i["result_slots"]) 
            df = pd.read_csv(service_dict[state_d['service_name']], encoding='utf-8-sig')
            if state_d['service_name']=="Messaging_1":
                for slot in opt_slot_i: data = df.loc[df[slot] != '無'].sample(replace=False, random_state=round(time.time()))
            else:
                data = df.sample(replace=False, random_state=round(time.time()))
            #(old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])         
            (usr_mat, sys_mat) = is_trans(value_list, intent_i)
            sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
            sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
            old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
            sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
            
        if sys_act == "OFFER_INTENT":   
            #while domain_index==0:
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index] 
            if not state_d['service_name'] in dialogues["services"]: dialogues["services"].append(state_d['service_name'])                     
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            (usr_mat, sys_mat) = is_trans(value_list, intent_i)
            
        if "AFFIRM_INTENT" in [usr_act, usr_act_inner]:
            goal_count += 1              
            req_slot_i = list(intent_i["required_slots"])
            opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
                if len(list(intent_i["optional_slots"].keys()))!=0 else []
            result_slots = list(intent_i["result_slots"])
            df = pd.read_csv(service_dict[state_d['service_name']], encoding='utf-8-sig')
            if state_d['service_name']=="Messaging_1":
                for slot in opt_slot_i: data = df.loc[df[slot] != '無'].sample(replace=False, random_state=round(time.time()))
            else:
                data = df.sample(replace=False, random_state=round(time.time())) 
            (old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])
            #(usr_mat, sys_mat) = is_trans(value_list, intent_i)

        if sys_act == "GOODBYE" or sys_act_inner == "GOODBYE":
            #usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
            usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data, state_d)
            sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [slots, result_slots], slot_values, state_d, intent_i, df)
            #------------------------------Record data------------------------------
            if  "NEGATE" in [usr_act, usr_act_inner] or "NEGATE_INTENT" in [usr_act, usr_act_inner]:
                usr_frames['state']['active_intent'] = "NONE"
            else:
                usr_frames['state']['active_intent'] = deepcopy(intent_i['name'])
            usr_frames['state']['requested_slots'] = deepcopy(requested_slots)
            usr_frames['state']['slot_values'] = deepcopy(slot_values)
            usr_frames["actions"] = deepcopy(list(usr_actions))
            sys_frames["actions"] = deepcopy(list(sys_actions))
            if service_call: sys_frames['service_call'] = service_call
            if service_results: sys_frames['service_results'] = service_results
            
            #print(usr_actions, slot_values)
            u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, state_d, './template/usr_template.json')
            s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, state_d, './template/sys_template.json', deepcopy(intent_i['name']))
            usr_turn["utterance"] = deepcopy(u_uttr)
            sys_turn["utterance"] = deepcopy(s_uttr)

            usr_frames['slots'] = deepcopy(u_slots_pos)
            sys_frames['slots'] = deepcopy(s_slots_pos)
            usr_turn["frames"] = [deepcopy(usr_frames)]
            sys_turn["frames"] = [deepcopy(sys_frames)]

            dialogues["turns"].append(deepcopy(usr_turn))
            dialogues["turns"].append(deepcopy(sys_turn))

            dialogs['Annotation(Actions)'].append(usr_annotations); dialogs['Template utterances'].append(u_uttr)
            dialogs['Annotation(Actions)'].append(sys_annotations); dialogs['Template utterances'].append(s_uttr)
            break    
    df = pd.DataFrame.from_dict(dialogs)
    df.to_csv(f'need_labeled/csv/{file_idx}.csv' ,index=False, encoding='utf-8-sig')
    with open(f'need_labeled/json/{file_idx}.json', 'w', encoding='utf-8-sig') as f:
        json.dump([dialogues], f, ensure_ascii=False, indent=5)
    return goal_count

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
    '''
    get_action_seq("./schema/messagewoz_schema.json")'''   
    num_multidomain = 0 
    for idx in range(1000):
        goal_count = generate_goal("./schema/messagewoz_schema.json", idx)
        if goal_count>1: num_multidomain += 1
        if num_multidomain%10==0:print(num_multidomain)
        #input()
    print(num_multidomain)    

    #print(usr_matrix)
    #print(sys_matrix)

    matrix_file = './matrix/matrix_weighted.npy'
    with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]

    non_matrix_file = './matrix/non_matrix_weighted.npy'
    with open(non_matrix_file, 'rb') as f:
        non_usr_matrix, non_sys_matrix, _, _ = [np.load(f) for _ in range(4)]
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
    