import json, random, os, time
import re
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
from data.analysis import is_transactional, analysis_schema
import data.build_db as db
import pandas as pd
from helper import get_query_text

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
    [0.   , 0.277, 0.115, 0.007, 0.   , 0.348, 0.   , 0.207, 0.046, 0.   , 0.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.322, 0.111, 0.006, 0.   , 0.321, 0.   , 0.192, 0.048, 0.   , 0.   ],
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
    [0.232, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.768],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
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
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.55 , 0.   , 0.   , 0.   , 0.   , 0.45 ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
    [0.   , 0.   , 0.55 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.45 ],
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

def get_usr_actions(acts, slots_, intent, data):
    value_list = analysis_schema("./schema/messagewoz_schema.json")
    slot_values = {}
    requested_slots = []
    slots = list(set(slots_[0] + slots_[1]))
    actions, annotations = [], []    
    for act in list(set(acts)):
        tmp_acts = []       
        if act == "none":continue       
        elif act == "INFORM":  
            
            ann_str = ''                   
            for slot in slots:  
                if data[slot].values[0] == "無":continue
                if not is_transactional(value_list, intent["name"]): 
                    if slot in ['message', 'event_content', 'subject', 'content']:     
                        data[slot].values[0] = get_query_text(data[slot].values[0])
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []} 
                act_dict["canonical_values"].append(data[slot].values[0])
                act_dict["values"].append(data[slot].values[0])   
                '''  
                if slot in ['copy_recipient', 'participant']:   
                    act_dict["canonical_values"] = data[slot].values[0].split('、')       
                    act_dict["values"] = data[slot].values[0].split('、') 
                else: 
                    act_dict["canonical_values"].append(data[slot].values[0])
                    act_dict["values"].append(data[slot].values[0])    
                '''  
                tmp_acts.append(act_dict)
                slot_values[slot] = [data[slot].values[0]]
                ann_str += f"{act}({slot}=〔{data[slot].values[0]}〕) "
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
            annotations.append(ann_str)
            requested_slots = list(slots)
        elif act == "INFORM_INTENT":
            act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
            act_dict["canonical_values"].append(intent["name"])
            act_dict["values"].append(intent["name"])
            tmp_acts.append(act_dict)
            annotations.append(f"{act}(" + f"{intent['name']})")
        else: 
            act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
            tmp_acts.append(act_dict)
            annotations.append(f"{act}()")
        if act == "REQUEST_ALTS":
            for slot in slots: slot_values[slot] = [data[slot].values[0]]
        elif act == "NEGATE":
            for slot in slots: slot_values[slot] = [data[slot].values[0]]
        actions += tmp_acts
        #print(act_dict, slots)
        #if act!='none': print(act, intent["name"])
    #print('usr', actions)
    #print('usr', slot_values, act)
    #input()
    return actions, slot_values, slots, 'U: ' + ' '.join(annotations), requested_slots

def get_sys_actions(acts, slots, slot_values, domain, intent, df):
    service_call = None
    service_results = []  
    columns = list(df.columns)              
    actions, annotations = [], []    
    for act in list(set(acts)):  
        tmp_acts = []  
        if act == "none":continue          
        elif act == "INFORM": 
            ann_str = ''
            #print('sys', slot_values, act)    
            req_results = db.find_result_slot(domain, slot_values, slots[0])
            #for slot, value in list(slot_values.items()):
            for k, v in zip(list(slots[0]), req_results[-1]):
                #if v == "無":continue
                act_dict = {"act": act, "canonical_values": [], "slot": k, "values": []}
                act_dict["canonical_values"].append(v)
                act_dict["values"].append(v)      
                tmp_acts.append(act_dict) 
                ann_str += f"{act}({k}=〔{v}〕) "
            annotations.append(ann_str)        
        elif act == "REQUEST":
            ann_str = ''
            for slot in slots[0]:
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                tmp_acts.append(act_dict) 
                ann_str += f"{act}({slot}) "
            annotations.append(ann_str) 
        elif act == "OFFER":
            ann_str = ''
            #print('sys', slot_values, act)
            service_call = {"method":intent["name"], "parameters":{}}
            results = db.find_result(domain, slot_values, slots[0])
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
                if value == "無":continue
                act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                act_dict["canonical_values"].append(value[0])
                act_dict["values"].append(value[0])      
                tmp_acts.append(act_dict)
                ann_str += f"{act}({slot}=〔{value[0]}〕) "
            annotations.append(ann_str)
        elif act == "INFORM_COUNT":  
            service_call = {"method":intent["name"], "parameters":{}}
            if slot_values: 
                results = db.find_result(domain, slot_values, slots[0])
                tmp_dict = {}
                for res in results:
                    for k, v in zip(list(columns), res):
                        tmp_dict[k] = v
                service_results.append(tmp_dict)
            for slot, value in slot_values.items(): service_call["parameters"][slot] = value[0]
            act_dict = {"act": act, "canonical_values": [len(results)], "slot": "count", "values": [len(results)]}
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
            for slot, value in set(list(slot_values.items())): service_call["parameters"][slot] = value[0]
        actions += tmp_acts
        #if act!='none': print(act, intent["name"])
    #print('sys', actions)
    return actions, 'S: ' + ' '.join(annotations), service_call, service_results

def print_actions(u_acts, s_acts, intent_i):
    print(u_acts[0], intent_i["name"])
    if u_acts[1]!='none': print(u_acts[1])
    print(s_acts[0])
    if s_acts[1]!='none': print(s_acts[1])

def usr_act2robot_utt(usr_actions, intent, domain):
    template, n = "", 0
    vocab = {'findmessage': '找一則訊息', 'sendmessage': '傳送一則訊息', 'message': '內容', 'contact_name': '聯絡人全名', 'app_name': '應用程式',\
        'event_date': '日期', 'event_time': '時間', 'event_location':'地點', 'event_name':'名稱', 'event_content':'內容',\
        'participant':'參加的人', 'available_start_time':'開始時間', 'available_end_time':'結束時間', 'messaging':'訊息',\
        'lookupevents':'找一個活動', 'getavailabletime':'找我什麼時候有空', 'addevent':'添加一個活動',\
        'recipient':'收件者', 'sender':'寄件者', 'subject':'主旨', 'content':'內容', 'copy_recipient':'副本收件者', 'mail':'信', 'calendar':'活動', \
        'sendmail':'寄一封信', 'findmail':'找一封信', 'REQUEST':'我想知道', 'THANK_YOU':'謝謝，已經達到我想要的事了。', 'INFORM_INTENT':'請幫我', 'INFORM':'好的，',\
        'AFFIRM':'對於這個', 'SELECT':'', 'NEGATE':'先這樣吧。', 'REQUEST_ALTS':'還有其他相關', 'GOODBYE':'', 'NEGATE_INTENT':'目前不需要。', 'AFFIRM_INTENT':'好的，麻煩你'}
    value_list = analysis_schema("./schema/messagewoz_schema.json")    
    #print(usr_actions)
    slots_pos = []
    for u_act in usr_actions:
        act, _, slot, value = list(u_act.values())
        #print(act)
        if value == "無":continue
        if n==0:
            if act == "INFORM":
                if is_transactional(value_list, intent["name"]):
                    template += vocab[act]
                else: template += vocab[act]+f"請用關鍵字詞查詢，"
            elif act == "REQUEST": template += f"{vocab[act]}這個{vocab[domain.replace('_1', '').lower()]}的"
            elif act=="REQUEST_ALTS":template += f"{vocab[act]}的{vocab[domain.replace('_1', '').lower()]}嗎？"
            else: template += vocab[act]
        
        if act == "INFORM_INTENT": template += f"{vocab[value[0].lower()]}。"
        elif act == "INFORM":
            if value == "無": template += f"沒有{vocab[slot.lower()]}，"
            else: template += f"{vocab[slot.lower()]}是「{value[0]}」，"
        elif act == "REQUEST": template += f"{vocab[slot.lower()]}、"
        elif act == "THANK_YOU": template += f"謝謝。"
        elif act == "AFFIRM": template += f"的資訊，我沒有任何問題。"
        elif act == "SELECT": template += f"好的。"
        elif act == "NEGATE": pass
        elif act == "REQUEST_ALTS": pass
        elif act == "GOODBYE": template += f"再會。"
        elif act == "NEGATE_INTENT": pass        
        elif act == "AFFIRM_INTENT": template += f"{vocab[intent['name'].lower()]}了。"
        n += 1
    template = list(template)
    if template[-1] == '，':template[-1] = '。'
    if template[-1] == '、':template[-1] = '。'
    template = ''.join(template)
    #print(template)
    slots_pos = find_slot(template, usr_actions)
    print("使用者:", template, '\n')
    return template, slots_pos

def find_slot(uttrance, acts):
    slots = []    
    for a in acts:        
        act, _, slot, value = list(a.values())
        if act == "INFORM": 
            if value[0]:
                for match in re.finditer(re.escape(value[0]), uttrance):
                    slots.append({"exclusive_end": match.end(), "slot": slot, "start": match.start()})
    return slots

def sys_act2robot_utt(sys_actions, intent, domain):
    template, n = "", 0
    vocab = {'findmessage': '找', 'sendmessage': '送出', 'message': '內容', 'contact_name': '聯絡人', 'app_name': '應用程式',\
        'event_date': '日期', 'event_time': '時間', 'event_location':'地點', 'event_name':'名稱', 'event_content':'內容',\
        'participant':'參加者', 'available_start_time':'開始時間', 'available_end_time':'結束時間', 'messaging':'訊息',\
        'getevents':'知道相關的活動', 'lookupevents':'找活動', 'getavailabletime':'有空的時間', 'addevent':'添加一個活動',\
        'recipient':'收件者', 'sender':'寄件者', 'subject':'主旨', 'content':'內容', 'copy_recipient':'副本收件者', 'mail':'信件', 'calendar':'活動', \
        'sendmail':'寄出', 'findmail':'找', 'REQUEST':'能告訴我這個', 'INFORM_COUNT':'我找到相關的', 'INFORM':'好的，',\
        'OFFER':'這是依據', 'CONFIRM':'請確認', 'NOTIFY_SUCCESS':'已將', 'REQ_MORE':'', 'GOODBYE':'', 'OFFER_INTENT':'您會想要', \
        'NOTIFY_FAILURE': '不好意思，無法達成您對於'}
    print(sys_actions)
    slots_pos = []
    for s_act in sys_actions:        
        act, _, slot, value = list(s_act.values())
        #print(act)
        #if value == "無":continue
        if n==0:
            if act == "OFFER_INTENT": template += f"{vocab[act]}{vocab[value[0].lower()]}"
            elif (act == "INFORM" or act == "REQUEST"):template += vocab[act]+vocab[domain.replace('_1', '').lower()]+'的'
            elif act != "GOODBYE" and act != "NOTIFY_FAILURE" and act != "INFORM" and act != "REQUEST" and act != "REQ_MORE": template += vocab[act]+vocab[domain.replace('_1', '').lower()]
            else: template += vocab[act]

        if act == "INFORM": 
            if value == "無": template += f"沒有{vocab[slot.lower()]}，"
            else: template += f"{vocab[slot.lower()]}是「{value[0]}」，"
        elif act == "REQUEST": template += f"{vocab[slot.lower()]}、"
        elif act == "OFFER": template += f"{vocab[slot.lower()]}、"
        elif act == "CONFIRM": template += f"{vocab[slot.lower()]}是{value[0]}，"
        elif act == "INFORM_COUNT": template += f"共有{str(value[0])}筆。"
        elif act == "NOTIFY_SUCCESS": template += f"成功{vocab[intent['name'].lower()]}。"
        elif act == "REQ_MORE": template += f"還需要其他服務嗎？"
        elif act == "GOODBYE": template += f"再會，掰掰。"
        elif act == "OFFER_INTENT": template += f"{vocab[domain.replace('_1', '').lower()]}嗎？"        
        elif act == "NOTIFY_FAILURE": template += f"{vocab[intent['name'].lower()]}需求。"  
        n += 1  
        #print(act, template)
    template = list(template)
    if template[-1] == '、' and act == "REQUEST":template[-1] = '嗎？'
    elif template[-1] == '、' and act == 'OFFER':template[-1] = f"所找到的相關{vocab[domain.replace('_1', '').lower()]}，結果如下："
    elif template[-1] == '，':template[-1] = '。'
    elif template[-1] == '、':template[-1] = '。'
    
    template = ''.join(template)
    #print(template)
    slots_pos = find_slot(template, sys_actions)
    print("助理:",template, '\n')
    return template, slots_pos

def get_action_seq(file_name):
    domain_index, max_num_goal, num_goal = 0, 3, 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8-sig").read())
    dialog_state_his, active_intents = [], []
    value_list = analysis_schema(file_name)
    domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0]     
    state_d = schemas[domain_index]               
    intent_i = random.choice(state_d["intents"])
    (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)
    usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]    
    sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
    u_acts = [u_act for u_act in [usr_act, usr_act_inner] if u_act!='none']
    s_acts = [s_act for s_act in [sys_act, sys_act_inner] if s_act!='none']
    while True:
        dialog_state_his.append(list(set(u_acts)))
        dialog_state_his.append(list(set(s_acts)))
        if sys_act == "GOODBYE" or sys_act_inner == "GOODBYE": break
            
        usr_idx = np.random.choice([i for i in range(len(usr_mat[old_sys_idx]))], 1, p=usr_mat[old_sys_idx])[0]
        usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
        old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)
        
        sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
        sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
        old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
        usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
        sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
        u_acts = [u_act for u_act in [usr_act, usr_act_inner] if u_act!='none']
        s_acts = [s_act for s_act in [sys_act, sys_act_inner] if s_act!='none']
        if usr_act == "INFORM_INTENT":
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0]             
            state_d = schemas[domain_index] 
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d) 
            (old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])
            _, _, _, (usr_mat, sys_mat) = gen_init_state(value_list, intent_i, [usr_idx, sys_idx])

        if "OFFER_INTENT" in [sys_act, sys_act_inner]: 
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                    
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            
        if "AFFIRM_INTENT" in [usr_act, usr_act_inner]:
            print([usr_act, usr_act_inner], [usr_idx, usr_idx_inner])
            print([sys_act, sys_act_inner], [sys_idx, sys_idx_inner])
            (old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])
            _, _, _, (usr_mat, sys_mat) = gen_init_state(value_list, intent_i, [usr_idx, sys_idx])

    pprint(dialog_state_his)
    return dialog_state_his

#def 

def generate_goal(file_name, file_idx):
    service_dict = {"Mail_1": 'data/csv/mail_entityies.csv', "Calendar_1": 'data/csv/events.csv', "Messaging_1": 'data/csv/message_entityies.csv'}
    dialogues = {"dialogue_id": file_idx, "services": [], "turns": []}
    domain_index, dialogs = 0, {'Annotation(Actions)':[], 'Template utterances':[]}
    schemas = json.loads(open(file_name, "r", encoding="utf-8-sig").read())
    #dialog_state_his = []
    value_list = analysis_schema(file_name)
    domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0]     
    state_d = schemas[domain_index]     
    df = pd.read_csv(service_dict[state_d['service_name']])
    data = df.sample(replace=True, random_state=round(time.time()))
    dialogues["services"].append(state_d['service_name'])                       
    intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
    req_slot_i = list(intent_i["required_slots"])
    opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
        if len(list(intent_i["optional_slots"].keys()))!=0 else []
    result_slots = list(intent_i["result_slots"])
    
    #initial state
    (usr_idx, sys_idx), (usr_idx_inner, sys_idx_inner), (old_usr_idx, old_sys_idx), (usr_mat, sys_mat) = gen_init_state(value_list, intent_i)
    usr_act, usr_act_inner = usr_acts[usr_idx], usr_acts[usr_idx_inner]
    sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
    while True:   
        usr_turn, sys_turn = {"frames":[], "speaker": "USER", "utterance":""}, {"frames":[], "speaker": "SYSTEM", "utterance":""}  
        usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
        #print_actions((usr_act, usr_act_inner), (sys_act, sys_act_inner), intent_i)
        
        usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data)
        sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [slots, result_slots], slot_values, state_d['service_name'], intent_i, df)
        
        #pprint(usr_actions)
        #input()
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
        u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, intent_i, state_d['service_name'])
        s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, intent_i, state_d['service_name'])
        usr_turn["utterance"] = deepcopy(u_uttr)
        sys_turn["utterance"] = deepcopy(s_uttr)

        usr_frames['slots'] = deepcopy(u_slots_pos)
        sys_frames['slots'] = deepcopy(s_slots_pos)

        usr_turn["frames"] = [deepcopy(usr_frames)]
        sys_turn["frames"] = [deepcopy(sys_frames)]   
        
        dialogues["turns"].append(deepcopy(usr_turn))
        dialogues["turns"].append(deepcopy(sys_turn))
        #pprint(dialogues)
        #input()
        dialogs['Annotation(Actions)'].append(usr_annotations); dialogs['Template utterances'].append(u_uttr)
        dialogs['Annotation(Actions)'].append(sys_annotations); dialogs['Template utterances'].append(s_uttr)
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
            df = pd.read_csv(service_dict[state_d['service_name']])
            data = df.sample(replace=True, random_state=round(time.time())) 
            if not state_d['service_name'] in dialogues["services"]: dialogues["services"].append(state_d['service_name'])                
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            req_slot_i = list(intent_i["required_slots"])
            opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
                if len(list(intent_i["optional_slots"].keys()))!=0 else []  
            result_slots = list(intent_i["result_slots"]) 
            #(old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])         
            (usr_mat, sys_mat) = is_trans(value_list, intent_i)
            sys_idx = np.random.choice([i for i in range(len(sys_mat[old_usr_idx]))], 1, p=sys_mat[old_usr_idx])[0]        
            sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
            old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
            sys_act, sys_act_inner = sys_acts[sys_idx], sys_acts[sys_idx_inner]
            
        if sys_act == "OFFER_INTENT":   
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index] 
            if not state_d['service_name'] in dialogues["services"]: dialogues["services"].append(state_d['service_name'])                     
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            (usr_mat, sys_mat) = is_trans(value_list, intent_i)
            
        if "AFFIRM_INTENT" in [usr_act, usr_act_inner]:
            df = pd.read_csv(service_dict[state_d['service_name']])
            data = df.sample(replace=True, random_state=round(time.time()))   
            req_slot_i = list(intent_i["required_slots"])
            opt_slot_i = random.choices(list(intent_i["optional_slots"].keys()), k=random.randint(1, len(list(intent_i["optional_slots"].keys()))))\
                if len(list(intent_i["optional_slots"].keys()))!=0 else []
            result_slots = list(intent_i["result_slots"])
            (old_usr_idx, old_sys_idx) = tuple([usr_idx, sys_idx])
            #(usr_mat, sys_mat) = is_trans(value_list, intent_i)

        if sys_act == "GOODBYE" or sys_act_inner == "GOODBYE":
            #usr_frames, sys_frames = {"actions":[], 'service':state_d['service_name'], 'slots': [], 'state':{}}, {"actions":[], 'service':state_d['service_name'], 'slots': []}
            usr_actions, slot_values, slots, usr_annotations, requested_slots = get_usr_actions([usr_act, usr_act_inner], [req_slot_i, opt_slot_i, result_slots], intent_i, data)
            sys_actions, sys_annotations, service_call, service_results = get_sys_actions([sys_act, sys_act_inner], [slots, result_slots], slot_values, state_d['service_name'], intent_i, df)
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
            u_uttr, u_slots_pos = usr_act2robot_utt(usr_actions, intent_i, state_d['service_name'])
            s_uttr, s_slots_pos = sys_act2robot_utt(sys_actions, intent_i, state_d['service_name'])
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
    '''get_action_seq("./schema/messagewoz_schema.json")
    '''
    for idx in range(1000):
        generate_goal("./schema/messagewoz_schema.json", idx)
        input()
        
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
    