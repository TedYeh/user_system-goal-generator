import json, random, os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy

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
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0.3, 0., 0., 0., 0., 0., 0.7, 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
]

usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOOD_BYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOOD_BYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]

def get_usr_act(usr_act, schemas, domain_index):    
    if usr_act != "none": 
        if usr_act == "INFORM_INTENT":
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                            
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            print('name', intent_i['name'])
            print('required_slots', intent_i["required_slots"])
            print('result_slots', intent_i["result_slots"])            
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
    optional_slot = None
    MAX_GOALS, goal_counter = 5, 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8").read())
    #initial state
    usr_idx = np.random.choice([i for i in range(len(usr_matrix[sys_idx]))], 1, p=usr_matrix[sys_idx])[0]
    usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
    old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)

    sys_idx = np.random.choice([i for i in range(len(sys_matrix[old_usr_idx]))], 1, p=sys_matrix[old_usr_idx])[0]    
    sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
    old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)
    #print(usr_idx, sys_idx)
    #print(usr_idx_inner, sys_idx_inner)
    #print(old_usr_idx, old_sys_idx)
    while True:          
        usr_act, sys_act = (usr_acts[usr_idx], sys_acts[sys_idx])
        usr_act_inner, sys_act_inner = (usr_acts[usr_idx_inner], sys_acts[sys_idx_inner])        
        #old_usr_idx, old_sys_idx = deepcopy(usr_idx_inner) if usr_act_inner!='none' else deepcopy(usr_idx), \
        #    deepcopy(sys_idx_inner) if sys_act_inner!='none' else deepcopy(sys_idx)

        if usr_act != "none": 
            if usr_act == "INFORM_INTENT":
                domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
                state_d = schemas[domain_index]                            
                intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
                slots_i = list(intent_i["required_slots"])
                #print(intent_i["required_slots"], intent_i["result_slots"], intent_i["optional_slots"])
                #print('name', intent_i['name'])
                #print('required_slots', intent_i["required_slots"])
                #print('result_slots', intent_i["result_slots"]) 
                print("usr", f"{usr_act}(", intent_i["name"], ")")
                
            elif usr_act == "INFORM":     
                if slots_i: print("usr", f"{usr_act}(", slots_i, "=)")
                else: 
                    optional_slot = np.random.choice(list(intent_i["optional_slots"].keys()), 1)[0]
                    print("usr", f"{usr_act}(", optional_slot, "=)")

            elif usr_act == "REQUEST":
                print("usr", f"{usr_act}(", slots_i, ")")
            else:
                print("usr", usr_act)

        if usr_act_inner != "none": 
            if usr_act_inner == "INFORM_INTENT":
                domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
                state_d = schemas[domain_index]                            
                intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
                slots_i = list(intent_i["required_slots"])
                #print('name', intent_i['name'])
                #print('required_slots', intent_i["required_slots"])
                #print('result_slots', intent_i["result_slots"])    
                print("usr", f"{usr_act_inner}(", intent_i["name"], ")")
            elif usr_act_inner == "INFORM":            
                if slots_i: print("usr", f"{usr_act_inner}(", slots_i, "=)")
                else: 
                    optional_slot = np.random.choice(list(intent_i["optional_slots"].keys()), 1)[0]
                    print("usr", f"{usr_act_inner}(", optional_slot, "=)")
            elif usr_act_inner == "REQUEST":
                print("usr", f"{usr_act_inner}(", slots_i, ")")
            else:
                print("usr", usr_act_inner)
        #domain_index = get_usr_act(usr_act, schemas, domain_index)
        #domain_index = get_usr_act(usr_act_inner, schemas, domain_index)
        print()
        if sys_act != "none":
            if  sys_act == "REQUEST":
                if slots_i: print("sys", f"{sys_act}(", slots_i, ")")
                elif optional_slot: print("sys", f"{sys_act}(", optional_slot, ")")
                else:
                    optional_slot = np.random.choice(list(intent_i["optional_slots"].keys()), 1)[0]
                    print("sys", f"{sys_act}(", optional_slot, ")")
            else:
                print("sys", f"{sys_act}")

        if sys_act_inner != "none": 
            print("sys", sys_act_inner)

        if sys_act == "GOOD_BYE" or sys_act_inner == "GOOD_BYE":
            break        
        print()
        usr_idx = np.random.choice([i for i in range(len(usr_matrix[old_sys_idx]))], 1, p=usr_matrix[old_sys_idx])[0]
        usr_idx_inner = np.random.choice([i for i in range(len(usr_inner_matrix[usr_idx]))], 1, p=usr_inner_matrix[usr_idx])[0]
        old_usr_idx = deepcopy(usr_idx_inner) if usr_acts[usr_idx_inner]!='none' else deepcopy(usr_idx)
        
        sys_idx = np.random.choice([i for i in range(len(sys_matrix[old_usr_idx]))], 1, p=sys_matrix[old_usr_idx])[0]        
        sys_idx_inner = np.random.choice([i for i in range(len(sys_inner_matrix[sys_idx]))], 1, p=sys_inner_matrix[sys_idx])[0]
        old_sys_idx = deepcopy(sys_idx_inner) if sys_acts[sys_idx_inner]!='none' else deepcopy(sys_idx)

def draw_matrix(matrix, select_acts=[], output_acts=[], labels=[], img_name='default'):
    
    fig, ax = plt.subplots()
    plt.title(img_name)
    im = ax.imshow(matrix, cmap='OrRd')
    # Major ticks
    ax.set_xticks(np.arange(0, len(matrix[0]), 1))
    ax.set_yticks(np.arange(0, len(matrix), 1))    

    # Labels for major ticks
    ax.set_xticklabels(output_acts, rotation=90, color="red" if labels[0]!='Client' else 'blue')
    ax.set_yticklabels(select_acts, rotation=0, color="red" if labels[1]!='Client' else 'blue')    
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, color="red" if labels[0]!='Client' else 'blue')
    ax.tick_params(axis="y", color="red" if labels[1]!='Client' else 'blue')
    ax.set_xlabel('decide action', color="red" if labels[0]!='Client' else 'blue')
    ax.set_ylabel('select action', color="red" if labels[1]!='Client' else 'blue')    

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
    plt.savefig(os.path.join('transistion matrix', img_name))
    plt.show()

if __name__ == "__main__":
    generate_goal("messagewoz_schema.json")

    '''
    usr_matrix = np.array(usr_matrix)    
    draw_matrix(usr_matrix, sys_acts[:-1], usr_acts[:-1], ['Client', 'Assistant'], 'Decide Client')

    sys_matrix = np.array(sys_matrix)
    draw_matrix(sys_matrix, usr_acts[:-1], sys_acts[:-1], ['Assistant', 'Client'], 'Decide Assistant')

    usr_inner_matrix = np.array(usr_inner_matrix)
    draw_matrix(usr_inner_matrix, usr_acts, usr_acts, ['Client', 'Client'], 'Continue Decide Client')

    sys_inner_matrix = np.array(sys_inner_matrix)
    draw_matrix(sys_inner_matrix, sys_acts, sys_acts, ['Assistant', 'Assistant'], 'Continue Decide Assistant')
    '''
    