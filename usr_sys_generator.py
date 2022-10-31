from email.utils import decode_rfc2231
import json, random, os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

domain_transition_matrix = [
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2]
]
usr_matrix = [
    [0., 0.2, 0.3, 0.5, 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.], 
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.8, 0., 0.2, 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
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
sys_inner_matrix = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
]
usr_inner_matrix = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0.5, 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
]

def get_usr_act(usr_act, schemas, domain_index):    
    if usr_act != "none": 
        if usr_act == "INFORM_INTENT":
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                            
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            pprint(intent_i['name'])
            pprint(intent_i["required_slots"])
            pprint(intent_i["result_slots"])            
            #input()
            print("usr", f"{usr_act}(", intent_i["name"], ")")
        else:print("usr", usr_act)
    return domain_index

def get_sys_act(sys_act, slots=[]):
    if sys_act != "none": 
        print("sys", sys_act)
    if sys_act == "GOOD_BYE": return True

def generate_goal(file_name):
    domain_index, usr_idx, sys_idx = 0, 3, 3
    u_act_list, s_act_list = [], []
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOOD_BYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOOD_BYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    
    MAX_GOALS, goal_counter = 5, 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8").read())
    #initial state
    usr_idx = np.random.choice([i for i in range(len(usr_matrix[sys_idx]))], 1, p=usr_matrix[sys_idx])[0]
    sys_idx = np.random.choice([i for i in range(len(sys_matrix[usr_idx]))], 1, p=sys_matrix[usr_idx])[0]
    while True:
        u_act_list, s_act_list = [], []           
        usr_act, sys_act = (usr_acts[usr_idx], sys_acts[sys_idx])
        usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
        sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
        usr_act_inner, sys_act_inner = (usr_acts[usr_idx_inner], sys_acts[sys_idx_inner])
        domain_index = get_usr_act(usr_act, schemas, domain_index)
        domain_index = get_usr_act(usr_act_inner, schemas, domain_index)
        print()
        
        is_bye = get_sys_act(sys_act, [])
        if is_bye:break
        is_bye = get_sys_act(sys_act_inner, [])
        print()
        if is_bye:break
        
        usr_idx = np.random.choice([i for i in range(len(usr_matrix[sys_idx]))], 1, p=usr_matrix[sys_idx])[0] \
            if sys_act_inner == "none" else np.random.choice([i for i in range(len(usr_matrix[sys_idx_inner]))], 1, p=usr_matrix[sys_idx_inner])[0]

        sys_idx = np.random.choice([i for i in range(len(sys_matrix[usr_idx]))], 1, p=sys_matrix[usr_idx])[0] \
            if usr_act_inner == "none" else np.random.choice([i for i in range(len(usr_matrix[usr_idx_inner]))], 1, p=usr_matrix[usr_idx_inner])[0]

if __name__ == "__main__":
    generate_goal("messagewoz_schema.json")
    usr_matrix = np.array(usr_matrix)
    print(usr_matrix)
    plt.matshow(usr_matrix)
    plt.show()

    sys_matrix = np.array(sys_matrix)
    plt.matshow(sys_matrix)
    plt.show()
    print(usr_matrix)
    '''
    #for file_name in os.listdir("train"):

        file_name = "dialogues_039.json"
        with open(os.path.join("train", file_name)) as f:
            for dialog in json.loads(f.read()):
                
                #if "Calendar_1" in dialog["services"]:
                #    print(file_name, dialog["dialogue_id"], dialog["services"])
                #    input()
                
                
                for turn in dialog["turns"]:
                    pprint(turn["speaker"])
                    pprint(turn["frames"][0]["actions"])   
                    print() 
                print("-"*60)
                input()
    '''