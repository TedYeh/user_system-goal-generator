import json, random, os, time
import re
from datetime import datetime
from helper import change_slot, find_slot_weight, get_uttr_list, find_slot
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
from data.analysis import is_transactional, analysis_schema
import data.build_db as db
import pandas as pd

#random.seed(datetime.now().timestamp())

DOMAIN = [
        [0.05, 0.5 , 0.45],
        [0.45, 0.05, 0.5 ],
        [0.5 , 0.45, 0.05]
]

MAX_GOAL = 3
U_MATRIX = np.array([
    [0.   , 0.126, 0.045, 0.486, 0.   , 0.138, 0.   , 0.   , 0.205, 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.323, 0.336, 0.   , 0.136, 0.   , 0.   , 0.205, 0.   , 0.   ],
    [0.   , 0.   , 0.187, 0.   , 0.   , 0.   , 0.813, 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.62 , 0.38 ],
    [0.   , 0.   , 0.   , 0.   , 0.898, 0.   , 0.056, 0.   , 0.046, 0.   , 0.   ]
])

S_MATRIX = np.array([
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

NON_U_MATRIX = np.array([
    [0.   , 0.126, 0.166, 0.038, 0.   , 0.401, 0.   , 0.205, 0.064, 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.277, 0.   , 0.007, 0.   , 0.463, 0.   , 0.207, 0.046, 0.   , 0.   ],
    [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.322, 0.   , 0.006, 0.   , 0.432, 0.   , 0.192, 0.048, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.177, 0.   , 0.   , 0.   , 0.823, 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.617, 0.383],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.908, 0.   , 0.092, 0.   , 0.   ]
])

NON_S_MATRIX = np.array([
    [0.   , 0.   , 0.547, 0.   , 0.   , 0.453, 0.   , 0.   , 0.   , 0.   ],
    [1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.615, 0.   , 0.   , 0.   , 0.385, 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.77 , 0.   , 0.   , 0.23 ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.367, 0.633, 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
    [0.   , 0.   , 0.563, 0.   , 0.   , 0.217, 0.   , 0.   , 0.   , 0.22 ],
    [0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
    [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]
])

U_C_MATRIX = np.array([
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

S_C_MATRIX = np.array([
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

usr_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"]
sys_acts = ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]

class messageSGD(object):
    def __init__(self, schema_file) -> None:
        self.service_dict = {"Mail_1": 'data/csv/mail_entityies.csv', "Calendar_1": 'data/csv/events.csv', "Messaging_1": 'data/csv/message_entityies_line.csv'}
        self.dial_idx, self.data_pos = 0, 0        
        self.schemas = json.loads(open(schema_file, "r", encoding="utf-8-sig").read())
        self.usr_templates = json.loads(open('./template/usr_template.json', 'r', encoding='utf-8').read())
        self.sys_templates = json.loads(open('./template/sys_template.json', 'r', encoding='utf-8').read())
        self.domain_index = np.random.choice([i for i in range(len(self.schemas))], 1, p=DOMAIN[0])[0]
        self.state_d = self.schemas[self.domain_index]
        self.intent_list = analysis_schema(schema_file)
        self.usr_dialog_state_his, self.sys_dialog_state_his = [], []
        #self.set_init_act()        
        self.goal_count = 0 
        self.dialog_type = {'multi':0, 'single':0}    
    
    def set_trans_mat(self):
        if is_transactional(self.intent_list, self.intent["name"]):
            self.usr_mat = U_MATRIX
            self.sys_mat = S_MATRIX                    
        else: 
            self.usr_mat = NON_U_MATRIX
            self.sys_mat = NON_S_MATRIX

    def set_init_act(self):
        self.set_trans_mat()
        self.results_log = []
        self.usr_idx, self.sys_idx = [3, 3]
        self.usr_idx = np.random.choice([i for i in range(len(self.usr_mat[self.sys_idx]))]\
            , 1, p=self.usr_mat[self.sys_idx])[0]
        self.usr_idx_c = np.random.choice([i for i in range(len(U_C_MATRIX[self.usr_idx]))]\
            , 1, p=U_C_MATRIX[self.usr_idx])[0]
        self.old_usr_idx = deepcopy(self.usr_idx_c) if usr_acts[self.usr_idx_c]!='none' else deepcopy(self.usr_idx)
        
        self.sys_idx = np.random.choice([i for i in range(len(self.sys_mat[self.old_usr_idx]))]\
            , 1, p=self.sys_mat[self.old_usr_idx])[0] 
        self.sys_idx_c = np.random.choice([i for i in range(len(S_C_MATRIX[self.sys_idx]))]\
            , 1, p=S_C_MATRIX[self.sys_idx])[0]
        self.old_sys_idx = deepcopy(self.sys_idx_c) if sys_acts[self.sys_idx_c]!='none' else deepcopy(self.sys_idx)
        
        self.usr_act, self.usr_act_c = usr_acts[self.usr_idx], usr_acts[self.usr_idx_c]
        self.sys_act, self.sys_act_c = sys_acts[self.sys_idx], sys_acts[self.sys_idx_c]

    def get_usr_act(self):
        self.usr_idx = np.random.choice([i for i in range(len(self.usr_mat[self.old_sys_idx]))]\
            , 1, p=self.usr_mat[self.old_sys_idx])[0]
        if self.goal_count >=3:
            if self.usr_idx == 2: self.usr_idx = 6
            elif self.usr_idx == 10: self.usr_idx = 9
        self.usr_idx_c = np.random.choice([i for i in range(len(U_C_MATRIX[self.usr_idx]))]\
            , 1, p=U_C_MATRIX[self.usr_idx])[0]
        self.old_usr_idx = deepcopy(self.usr_idx_c) if usr_acts[self.usr_idx_c]!='none' else deepcopy(self.usr_idx)
        self.usr_act, self.usr_act_c = usr_acts[self.usr_idx], usr_acts[self.usr_idx_c]

    def get_sys_act(self):     
        self.sys_idx = np.random.choice([i for i in range(len(self.sys_mat[self.old_usr_idx]))]\
            , 1, p=self.sys_mat[self.old_usr_idx])[0]
        self.sys_idx_c = np.random.choice([i for i in range(len(S_C_MATRIX[self.sys_idx]))]\
            , 1, p=S_C_MATRIX[self.sys_idx])[0]
        self.old_sys_idx = deepcopy(self.sys_idx_c) if sys_acts[self.sys_idx_c]!='none' else deepcopy(self.sys_idx)
        self.sys_act, self.sys_act_c = sys_acts[self.sys_idx], sys_acts[self.sys_idx_c]

    def set_dialog_record(self):
        self.dialogues = {"dialogue_id": self.dial_idx, "services": [], "turns": []}
        self.dialogs = {'Annotation(Actions)':[], 'Template utterances':[]} 
        self.dialogues["services"].append(self.state_d['service_name'])       

    def set_usr_init_turn(self):
        self.usr_turn = {"frames":[], "speaker": "USER", "utterance":""}
        self.usr_frames = {"actions":[], 'service':self.state_d['service_name'], 'slots': [], 'state':{}} 
    
    def set_sys_init_turn(self):
        self.sys_turn = {"frames":[], "speaker": "SYSTEM", "utterance":""}   
        self.sys_frames = {"actions":[], 'service':self.state_d['service_name'], 'slots': []}

    def record_usr_dialog(self):
        self.usr_frames['state']['active_intent'] = deepcopy(self.intent['name'])
        self.usr_frames['state']['requested_slots'] = deepcopy(self.requested_slots)
        self.usr_frames['state']['slot_values'] = self.usr_state
        self.usr_frames["actions"] = deepcopy(list(self.usr_actions))
        self.usr_turn["utterance"] = deepcopy(self.u_uttr)
        self.usr_frames['slots'] = deepcopy(self.u_slots_pos)
        self.usr_turn["frames"] = [deepcopy(self.usr_frames)]
        self.dialogues["turns"].append(deepcopy(self.usr_turn))
        self.dialogs['Annotation(Actions)'].append(self.usr_annotations) 
        self.dialogs['Template utterances'].append(self.u_uttr)

    def record_sys_dialog(self):
        self.sys_frames["actions"] = deepcopy(list(self.sys_actions))
        if self.service_call: self.sys_frames['service_call'] = self.service_call
        self.sys_frames['service_results'] = self.service_results
        self.sys_frames['slots'] = deepcopy(self.s_slots_pos)
        self.sys_turn["frames"] = [deepcopy(self.sys_frames)]
        self.sys_turn["utterance"] = deepcopy(self.s_uttr)
        self.dialogues["turns"].append(deepcopy(self.sys_turn))
        self.dialogs['Annotation(Actions)'].append(self.sys_annotations)
        self.dialogs['Template utterances'].append(self.s_uttr)

    def decide_intent_slots(self):
        self.intent = random.choice(self.state_d["intents"]) #select intent from domain_i(state_d)
        self.req_slot = list(self.intent["required_slots"])
        self.opt_slot = random.choices(list(self.intent["optional_slots"].keys()), k=random.randint(1, len(list(self.intent["optional_slots"].keys()))))\
            if len(list(self.intent["optional_slots"].keys()))!=0 else []
        self.result_slots = list(self.intent["result_slots"])
        self.slot_values, self.data_pos = {}, 0

    def sample_data(self):
        self.df = pd.read_csv(self.service_dict[self.state_d['service_name']], encoding='utf-8-sig')
        if self.state_d['service_name']=="Messaging_1":
            for slot in self.opt_slot: self.data = self.df.loc[self.df[slot] != '無'].sample(replace=False).reset_index(drop=True)
        else: self.data = self.df.sample(replace=False).reset_index(drop=True)#round(time.time())

    def get_usr_actions(self, acts, slots_):
        self.requested_slots = []
        slots = sorted(list(set(slots_[0] + slots_[1])), key=lambda x:find_slot_weight(self.state_d['slots'], x), reverse=True)
        self.usr_actions, self.usr_annotations = [], []    
        for act in list(set(acts)):
            tmp_acts = []       
            if act == "none":continue       
            elif act == "INFORM":              
                ann_str = ''                   
                for slot in slots:  
                    act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []} 
                    values = change_slot(slot, self.data[slot].values[0])
                    act_dict["canonical_values"].append(self.data[slot].values[0])
                    act_dict["values"].append(values) 
                    tmp_acts.append(act_dict)
                    self.slot_values[slot] = [values]
                    ann_str += f"{act}({slot}=〔{values}〕) "
                self.usr_annotations.append(ann_str)                 
            elif act == "REQUEST":
                #print(slots, slots_[2])
                for slot in slots: 
                    if slot in slots_[2]: slots_[2].remove(slot)
                if slots_[2]: slots = list(set(random.choices(list(slots_[2]), k=random.randint(1, len(list(slots_[2]))))))
                ann_str = ''
                for slot in slots:
                    act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                    #self.slot_values[slot] = [self.data[slot].values[0]]
                    tmp_acts.append(act_dict)  
                    ann_str += f"{act}({slot}) "
                    if slot in slots_[2]: slots_[2].remove(slot)
                self.usr_annotations.append(ann_str)
                self.requested_slots = list(slots)
            elif act == "INFORM_INTENT":
                act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
                act_dict["canonical_values"].append(self.intent["name"])
                act_dict["values"].append(self.intent["name"])
                tmp_acts.append(act_dict)
                self.usr_annotations.append(f"{act}(" + f"{self.intent['name']})")
                #for slot in slots:  
                #    self.slot_values[slot] = [self.data[slot].values[0]]
            elif act=="AFFIRM_INTENT":
                act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                tmp_acts.append(act_dict)
                self.usr_annotations.append(f"{act}()")
                #for slot in slots:  
                #    self.slot_values[slot] = [self.data[slot].values[0]]
            elif act=="AFFIRM":
                act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                tmp_acts.append(act_dict)
                self.usr_annotations.append(f"{act}()")
                #for slot in slots:  
                #    self.slot_values[slot] = [self.data[slot].values[0]]
            else: 
                act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                tmp_acts.append(act_dict)
                self.usr_annotations.append(f"{act}()")
            #if act == "REQUEST_ALTS":
            #    for slot in slots: self.slot_values[slot] = [self.data[slot].values[0]]
            #elif act == "NEGATE":
            #    for slot in slots: self.slot_values[slot] = [self.data[slot].values[0]]
            self.usr_actions += deepcopy(tmp_acts)
        self.usr_annotations = 'U: ' + ' '.join(self.usr_annotations)
        return slots

    def is_in_log(self, results):
        for result in results:
            if not result:continue
            if list(result.values()) in self.results_log:
                return True
            self.results_log.append(list(result.values()))
        return False

    def get_sys_actions(self, acts, slots):
        domain = self.state_d['service_name']
        self.service_call = None
        self.service_results = []  
        columns = list(self.df.columns)              
        self.sys_actions, self.sys_annotations = [], []  
        for act in list(set(acts)):  
            tmp_acts = []  
            if act == "none":continue          
            elif act == "INFORM": 
                ann_str = ''   
                req_results = db.find_result_slot(domain, self.slot_values, slots[0])
                for k, v in zip(list(slots[0]), req_results[-1]):
                    act_dict = {"act": act, "canonical_values": [], "slot": k, "values": []}
                    act_dict["canonical_values"].append(v)
                    act_dict["values"].append(v)      
                    tmp_acts.append(act_dict) 
                    ann_str += f"{act}({k}=〔{v}〕) "
                self.sys_annotations.append(ann_str)        
            elif act == "REQUEST":
                ann_str = ''
                for slot in sorted(slots[0], key=lambda x:find_slot_weight(self.state_d['slots'], x), reverse=True):
                    act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                    tmp_acts.append(act_dict) 
                    ann_str += f"{act}({slot}) "
                self.sys_annotations.append(ann_str) 
            elif act == "OFFER":
                '''
                if self.sys_idx == 2 or self.sys_idx == 5: 
                    if self.is_in_result(): self.sys_idx = 9 
                '''    
                ann_str = ''
                self.service_call = {"method":self.intent["name"], "parameters":{}}
                results = db.find_result(domain, self.slot_values)[self.data_pos:self.data_pos+5] #, slots[0]
                if self.slot_values:                 
                    tmp_dict = {}
                    for res in results:
                        for k, v in zip(list(columns), res):
                            tmp_dict[k] = v
                    if tmp_dict:self.service_results.append(deepcopy(tmp_dict))
                if (not self.is_in_log(self.service_results)) and len(results)>=1:
                    for slot, value in self.slot_values.items():
                        act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                        act_dict["canonical_values"].append(value[0])
                        act_dict["values"].append(value[0])     
                        self.service_call["parameters"][slot] = value
                        if not act_dict in tmp_acts: tmp_acts.append(act_dict)
                        ann_str += f"{act}({slot}=〔{value[0]}〕) "
                    self.sys_annotations.append(ann_str)
                else:
                    act = "NOTIFY_FAILURE"
                    acts[-1] = "REQ_MORE"
                    act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                    if not f"{act}()" in self.sys_annotations: 
                        tmp_acts.append(act_dict)
                        #tmp_acts.append({"act": "REQ_MORE", "canonical_values": [], "slot": "", "values": []})
                    self.sys_annotations.append(f"{act}()")   
                    self.old_sys_idx = 7
                    self.sys_act = "NOTIFY_FAILURE"           
            elif act == "CONFIRM": 
                ann_str = ''
                for slot, value in list(self.slot_values.items()):
                    #if value == "無":continue
                    act_dict = {"act": act, "canonical_values": [], "slot": slot, "values": []}
                    act_dict["canonical_values"].append(value[0])
                    act_dict["values"].append(value[0])      
                    tmp_acts.append(act_dict)
                    ann_str += f"{act}({slot}=〔{value[0]}〕) "
                self.sys_annotations.append(ann_str)
            elif act == "INFORM_COUNT":  
                self.service_call = {"method":self.intent["name"], "parameters":{}}  
                results = db.find_result(domain, self.slot_values) 
                if len(results)>0:
                    for slot, value in self.slot_values.items(): self.service_call["parameters"][slot] = value[0]
                    act_dict = {"act": act, "canonical_values": [str(len(results))], "slot": "count", "values": [str(len(results))]}
                    tmp_acts.append(act_dict)
                    self.sys_annotations.append(f"{act}(" + f"{len(results)})")
                else:
                    act = "NOTIFY_FAILURE"
                    acts[-1] = "REQ_MORE"
                    act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                    if not f"{act}()" in self.sys_annotations: 
                        tmp_acts.append(act_dict)
                        #tmp_acts.append({"act": "REQ_MORE", "canonical_values": [], "slot": "", "values": []})
                    self.sys_annotations.append(f"{act}()")   
                    self.old_sys_idx = 7
                    self.sys_act = "NOTIFY_FAILURE"
            elif act == "OFFER_INTENT":
                act_dict = {"act": act, "canonical_values": [], "slot": "intent", "values": []}
                act_dict["canonical_values"].append(self.intent["name"])
                act_dict["values"].append(self.intent["name"])
                tmp_acts.append(act_dict)
                self.sys_annotations.append(f"{act}(" + f"{self.intent['name']})")
            else: 
                act_dict = {"act": act, "canonical_values": [], "slot": "", "values": []}
                tmp_acts.append(act_dict)
                self.sys_annotations.append(f"{act}()")
            if act == "NOTIFY_SUCCESS" or act == "NOTIFY_FAILURE":
                self.service_call = {"method":self.intent["name"], "parameters":{}}
                for slot, value in list(self.slot_values.items()): self.service_call["parameters"][slot] = value[0]
            self.sys_actions += deepcopy(tmp_acts)
        self.sys_annotations = 'S: ' + ' '.join(self.sys_annotations)

    def usr_act2robot_utt(self):
        if list(self.usr_actions[-1].values())[0] in ["INFORM_INTENT", "NEGATE", "SELECT"]:
            self.usr_actions[0], self.usr_actions[-1] = self.usr_actions[-1], self.usr_actions[0]
        domain = self.state_d['service_name']
        slots, slot_act_dict = [], {}        
        for u_act in self.usr_actions:        
            act, _, slot, value = list(u_act.values()) 
            if not act in slot_act_dict: slot_act_dict[act]={}       
            if act == "INFORM_INTENT":
                slot_act_dict[act][value[0]] = value[0]
            else:
                if '無' in value: slot += '*無'  
                if slot == '': slot='none' 
                slot_act_dict[act][slot] = value[0] if value else 'none'   
        uttr_templates = get_uttr_list(slot_act_dict, self.state_d, self.usr_templates[domain])
        self.u_uttr = '，'.join(uttr_templates)
        self.u_slots_pos = find_slot(self.u_uttr, self.usr_actions)
        print('使用者：', self.u_uttr)

    def sys_act2robot_utt(self, active_intent=''):
        if len(self.sys_actions) == 0:print(self.sys_actions)
        if list(self.sys_actions[-1].values())[0] in ["NOTIFY_FAILURE"]:
            self.usr_actions[0], self.usr_actions[-1] = self.usr_actions[-1], self.usr_actions[0]
        domain = self.state_d['service_name']
        slots, slot_act_dict = [], {}
        for s_act in self.sys_actions:        
            act, _, slot, value = list(s_act.values())
            if not act in slot_act_dict: slot_act_dict[act]={}  
            if act=="OFFER_INTENT":
                slot_act_dict[act][value[0]] = value[0]
            elif act in ["NOTIFY_SUCCESS", "NOTIFY_FAILURE"]:
                slot_act_dict[act][self.intent["name"]] = active_intent if not active_intent=="" else "none"#
            else:
                if '無' in value: slot += '*無'
                if slot == '': slot='none'
                slot_act_dict[act][slot] = value[0] if value else 'none' 
        uttr_templates = get_uttr_list(slot_act_dict, self.state_d, self.sys_templates[domain])
        self.s_uttr = '，'.join(uttr_templates)
        self.s_slots_pos = find_slot(self.s_uttr, self.sys_actions)
        if self.s_uttr == '':
            print(self.sys_actions, slot_act_dict)
            input()
        print('助理：', self.s_uttr)

    def generate_goal(self):
        self.set_dialog_record()                       
        self.decide_intent_slots()
        self.sample_data()
        self.set_init_act()
        while True:
            self.set_usr_init_turn()
            self.set_sys_init_turn()
            if ("REQUEST" in [self.sys_act, self.sys_act_c]):
                u_acts, u_slots = [self.usr_act, self.usr_act_c], [self.req_slot, self.opt_slot, self.result_slots]
                slots = self.get_usr_actions(u_acts, u_slots)
                self.usr_state = {}
                for u_action in self.usr_actions: self.usr_dialog_state_his.append(list(u_action.keys())[0:2])
                self.usr_act2robot_utt()
                self.record_usr_dialog()
                self.get_usr_act()  
                slots_ = list(set(self.req_slot + self.opt_slot))
                while len(slots_)!=0:
                    self.set_usr_init_turn()
                    self.set_sys_init_turn()
                    select_slot = list(set(random.choices(slots_, k=random.randint(1, 3)))) #最多詢問三項資訊
                    s_acts, s_slots = [self.sys_act, self.sys_act_c], [select_slot, self.result_slots]
                    self.get_sys_actions(s_acts, s_slots)
                    for s_action in self.sys_actions: self.sys_dialog_state_his.append(list(s_action.keys())[0:2])
                    u_acts, u_slots = [self.usr_act, self.usr_act_c], [select_slot, select_slot, self.result_slots]
                    _ = self.get_usr_actions(u_acts, u_slots)
                    for u_action in self.usr_actions: self.usr_dialog_state_his.append(list(u_action.keys())[0:2])
                    self.sys_act2robot_utt(deepcopy(self.intent['name']))
                    self.usr_act2robot_utt()    
                    self.record_sys_dialog()
                    self.usr_state = deepcopy(self.slot_values)
                    self.record_usr_dialog()
                    for s in select_slot: slots_.remove(s)
                self.set_sys_init_turn()
                self.old_usr_idx = deepcopy(self.usr_idx_c) if usr_acts[self.usr_idx_c]!='none' else deepcopy(self.usr_idx)
                self.get_sys_act()
                s_acts, s_slots = [self.sys_act, self.sys_act_c], [list(set(self.req_slot+self.opt_slot)), self.result_slots]
                self.get_sys_actions(s_acts, s_slots)    
                self.sys_act2robot_utt()
                self.record_sys_dialog()
                self.get_usr_act() 
                self.set_usr_init_turn()
                self.set_sys_init_turn()   
                self.get_sys_act()
            u_acts, u_slots = [self.usr_act, self.usr_act_c], [self.req_slot, self.opt_slot, self.result_slots]
            slots = self.get_usr_actions(u_acts, u_slots)
            s_acts, s_slots = [self.sys_act, self.sys_act_c], [slots, self.result_slots]
            self.get_sys_actions(s_acts, s_slots)
            if  "NEGATE" in [self.usr_act, self.usr_act_c] or "NEGATE_INTENT" in [self.usr_act, self.usr_act_c]:
                self.usr_frames['state']['active_intent'] = "NONE"
            self.usr_act2robot_utt()
            self.sys_act2robot_utt(deepcopy(self.intent['name']))   
            self.usr_state = deepcopy(self.slot_values)
            self.record_usr_dialog()             
            self.record_sys_dialog()
            self.get_usr_act()
            self.get_sys_act()
            if "REQUEST_ALTS" in [self.usr_act, self.usr_act_c]:
                self.data_pos += 5
                self.sample_data()
            if "INFORM_INTENT" in [self.usr_act, self.usr_act_c]:
                self.usr_state = {}
                self.slot_values = {}
                self.goal_count += 1
                self.domain_index = np.random.choice([i for i in range(len(self.schemas))]\
                    , 1, p=DOMAIN[self.domain_index])[0]             
                self.state_d = self.schemas[self.domain_index]                 
                if not self.state_d['service_name'] in self.dialogues["services"]: 
                    self.dialogues["services"].append(self.state_d['service_name'])                
                self.intent = random.choice(self.state_d["intents"]) #select intent from domain_i(state_d)
                self.req_slot = list(self.intent["required_slots"])
                self.opt_slot = random.choices(list(self.intent["optional_slots"].keys())\
                    , k=random.randint(1, len(list(self.intent["optional_slots"].keys()))))\
                    if len(list(self.intent["optional_slots"].keys()))!=0 else []  
                self.result_slots = list(self.intent["result_slots"]) 
                self.sample_data()        
                self.set_trans_mat()
                self.get_sys_act()
                
            if self.sys_act == "OFFER_INTENT":   
                self.domain_index = np.random.choice([i for i in range(len(self.schemas))]\
                    , 1, p=DOMAIN[self.domain_index])[0]             
                self.state_d = self.schemas[self.domain_index]       
                self.intent = random.choice(self.state_d["intents"]) #select intent from domain_i(state_d)
                self.set_trans_mat()
                
            if "AFFIRM_INTENT" in [self.usr_act, self.usr_act_c]:
                self.goal_count += 1           
                if not self.state_d['service_name'] in self.dialogues["services"]: 
                    self.dialogues["services"].append(self.state_d['service_name'])  
                self.req_slot = list(self.intent["required_slots"])
                self.opt_slot = random.choices(list(self.intent["optional_slots"].keys())\
                    , k=random.randint(1, len(list(self.intent["optional_slots"].keys()))))\
                    if len(list(self.intent["optional_slots"].keys()))!=0 else []  
                self.result_slots = list(self.intent["result_slots"]) 
                self.sample_data()
                (self.old_usr_idx, self.old_sys_idx) = tuple([self.usr_idx, self.sys_idx]) 

            if self.sys_act == "GOODBYE" or self.sys_act_c == "GOODBYE":
                if self.goal_count > 1: self.dialog_type['multi']+=1
                else: self.dialog_type['single']+=1
                u_acts, u_slots = [self.usr_act, self.usr_act_c], [self.req_slot, self.opt_slot, self.result_slots]
                slots = self.get_usr_actions(u_acts, u_slots)
                s_acts, s_slots = [self.sys_act, self.sys_act_c], [slots, self.result_slots]
                self.get_sys_actions(s_acts, s_slots)
                #------------------------------Record data------------------------------
                if  "NEGATE" in [self.usr_act, self.usr_act_c] or "NEGATE_INTENT" in [self.usr_act, self.usr_act_c]:
                    self.usr_frames['state']['active_intent'] = "NONE"
                self.usr_act2robot_utt()
                self.sys_act2robot_utt(deepcopy(self.intent['name']))  
                self.usr_state = deepcopy(self.slot_values)
                self.record_usr_dialog()                  
                self.record_sys_dialog()   
                self.goal_count = 0    
                self.usr_state = {}         
                break  
    
    def run(self):
        for file_idx in range(500):
            self.dial_idx = int(file_idx)
            self.generate_goal()
            with open(f'need_labeled/new_json/{file_idx}.json', 'w', encoding='utf-8-sig') as f:
                json.dump([self.dialogues], f, ensure_ascii=False, indent=5)
            #self.slot_values = {}    
            print()
        print(self.dialog_type)

if __name__ == "__main__":
    goal_agent = messageSGD("./schema/messagewoz_schema.json")
    goal_agent.run()
