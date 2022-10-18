import json, random, os
import numpy as np
from pprint import pprint

def generate_goal(file_name):
    domain_index, usr_idx, sys_idx = 0, 3, 3
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU"], ["INFORM", "REQUEST", "OFFER", "GOOD_BYE"]
    domain_transition_matrix = [
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2]
    ]
    usr_matrix = [
        [0., 0.2, 0.3, 0.5],
        [1., 0., 0., 0.],
        [0., 0.2, 0., 0.8],
        [0., 0., 1., 0.]
    ]
    sys_matrix = [
        [0.5, 0., 0.5, 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0],
        [0., 0., 0., 1.]
    ]
    MAX_GOALS, goal_counter = 5, 0
    schemas = json.loads(open(file_name, "r", encoding="utf-8").read())
    #initial state
    usr_idx = np.random.choice([i for i in range(len(usr_matrix[sys_idx]))], 1, p=usr_matrix[sys_idx])[0]
    sys_idx = np.random.choice([i for i in range(len(sys_matrix[usr_idx]))], 1, p=sys_matrix[usr_idx])[0]
    while True:           
        usr_act, sys_act = (usr_acts[usr_idx], sys_acts[sys_idx])
        # select from slots of intent_i & generate user D.A. label
        if usr_act == "INFORM_INTENT":
            domain_index = np.random.choice([i for i in range(len(schemas))], 1, p=domain_transition_matrix[domain_index])[0] 
            state_d = schemas[domain_index]                
            intent_i = random.choice(state_d["intents"]) #select intent from domain_i(state_d)
            print("usr", f"{usr_act}(", intent_i["name"], ")")
        else:
            if len(intent_i["required_slots"]) == 0:slot = random.choice(list(intent_i["optional_slots"].keys()))
            print("usr", f"{usr_act}(",")")
        print()
        if sys_act == "GOOD_BYE":
            print("sys", sys_act)
            break
        else:
            print("sys", f"{sys_act}(",")")
        print()
        usr_idx = np.random.choice([i for i in range(len(usr_matrix[sys_idx]))], 1, p=usr_matrix[sys_idx])[0]
        sys_idx = np.random.choice([i for i in range(len(sys_matrix[usr_idx]))], 1, p=sys_matrix[usr_idx])[0]
        '''
        print("INFORM_INTENT(", intent_i["name"], ")")
        print()
        if len(intent_i["required_slots"]) == 0:
            slot = random.choice(list(intent_i["optional_slots"].keys()))
            print("REQUEST(", slot, ")")
            print()
            print("INFORM(", slot, "=)")
        else:
            for slot in intent_i["required_slots"]:
                print("REQUEST(", slot, ")")
            print()
            for slot in intent_i["required_slots"]:
                print("INFORM(", slot, "=)")
        print()
        #print('-'*20)
        '''
        #goal_counter += 1
        #input()
    #for schema in schemas:
    #    pprint(schema)
    #    input()

if __name__ == "__main__":
    generate_goal("messagewoz_schema.json")
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