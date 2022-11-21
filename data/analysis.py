import os, json
import numpy as np
from pprint import pprint
import pandas as pd

def get_action_times(paths):
    is_start, is_in_start = False, False
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    usr_index, sys_index = 0, 0 
    usr_in_index, sys_in_index = -1, -1 
    usr_matrix = np.zeros((len(sys_acts)-1, len(usr_acts)-1))
    sys_matrix = np.zeros((len(usr_acts)-1, len(sys_acts)-1))

    usr_inner_matrix = np.zeros((len(usr_acts), len(usr_acts)))
    sys_inner_matrix = np.zeros((len(sys_acts), len(sys_acts)))
    
    action_times = {}
    for path in paths:
        file_path = os.path.join('sgd', path)
        for d_file in os.listdir(file_path)[:-1]:
            file_name = os.path.join(file_path, d_file)
            dialogs = json.loads(open(file_name, encoding='utf-8').read())
            for dialog in dialogs:
                for turn in dialog["turns"]:
                    for frame in turn["frames"]:  
                        if len(frame["actions"]) > 1:  
                            usr_in_index, sys_in_index = -1, -1                     
                            for act in frame["actions"]:
                                #print(turn['speaker']+'-'+act['act'])
                                if not turn['speaker']+'-'+act['act'] in action_times:action_times[turn['speaker']+'-'+act['act']] = 1
                                else:action_times[turn['speaker']+'-'+act['act']] += 1
                        else:
                            act = frame["actions"][0]
                            if not turn['speaker']+'-'+act['act'] in action_times:action_times[turn['speaker']+'-'+act['act']] = 1
                            else:action_times[turn['speaker']+'-'+act['act']] += 1                            
                        #act = frame["actions"][0]                               
                        if not is_start: 
                            is_start = True
                            if turn['speaker'] == 'USER':usr_index = usr_acts.index(act['act'])
                            else:sys_index = sys_acts.index(act['act'])
                        else:
                            if turn['speaker'] == 'USER':
                                usr_index = usr_acts.index(act['act'])
                                usr_matrix[sys_index, usr_index] += 1
                            else:
                                sys_index = sys_acts.index(act['act'])
                                sys_matrix[usr_index, sys_index] += 1
                            
                is_start = False
    pprint(action_times)
    df = pd.DataFrame({'Action_name':action_times.keys(), "Amount":action_times.values()})
    df.to_csv('times.csv', index=False)
    print("usr")
    print(usr_matrix)
    print("sys")
    print(sys_matrix)
    

    print("usr inner")
    print(usr_inner_matrix)
    print("sys inner")
    print(sys_inner_matrix)
    with open('./matrix.npy', 'wb') as f:
        np.save(f, usr_matrix)
        np.save(f, sys_matrix)
        np.save(f, usr_inner_matrix)
        np.save(f, sys_inner_matrix)

def get_weighted_matrix(matrix_file):   
    with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]
        usr_matrix[3, 2] = 1.

    with open('./matrix_weighted.npy', 'wb') as f:
        for matrix in [usr_matrix, sys_matrix]:
            for i in range(matrix.shape[0]):
                if not sum(matrix[i])==0:
                    matrix[i] /= sum(matrix[i])
            print(np.around(matrix, decimals=3))
            np.save(f, matrix)
        for matrix in [usr_inner_matrix, sys_inner_matrix]:
            for i in range(matrix.shape[0]):
                if not sum(matrix[i])==0:
                    matrix[i] /= sum(matrix[i])
                else: matrix[i][-1] = 1.
            print(np.around(matrix, decimals=3))
            np.save(f, matrix)
                
def analysis_ptt(path):
    comments_amount = 0
    file_path = os.path.join('./', path)
    for d_file in os.listdir(file_path)[:-1]:
        file_name = os.path.join(file_path, d_file)
        comments = json.loads(open(file_name, encoding='utf-8').read())
        for comment in comments:
            comments_amount += len(comment)
    print(comments_amount)          

if __name__=="__main__":
    #paths = ['train', 'test', 'dev']
    #get_action_times(paths)
    #get_weighted_matrix("./matrix.npy")
    path = 'ptt\\data\\source_replies\\reply'
    analysis_ptt(path)