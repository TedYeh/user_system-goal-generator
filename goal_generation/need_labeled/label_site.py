from bs4 import BeautifulSoup
import requests, json, random
import os, time, re
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from flask import Flask, flash, request, redirect, url_for, render_template, session

app = Flask(__name__)
app.config['SECRET_KEY'] = b'>\x89k\xff.t{\xed\xc0\x8c^E\x81A\xe7\xb6'

def find_slot(uttrance, acts):
    slots = []    
    for a in acts:        
        act, _, slot, value = list(a.values())
        if act in ["INFORM", "OFFER", "CONFIRM"]: 
            if value[0]:
                for match in re.finditer(re.escape(value[0]), uttrance):
                    slots.append({"exclusive_end": match.end(), "slot": slot, "start": match.start()})
    return slots

def get_act_anntation_format(actions):
    annotations = []
    for action in actions:
        act, slot, values = action['act'], action['slot'], action['values']
        if act in ["INFORM", "OFFER", "CONFIRM"]: annotations.append(f"{act}({slot}=〔{values[0]}〕)")                 
        elif act == "REQUEST": annotations.append(f"{act}({slot})")
        elif act in ["INFORM_INTENT", "OFFER_INTENT", "INFORM_COUNT"]: annotations.append(f"{act}({values[0]})")
        else: annotations.append(f"{act}()")
    return annotations

def get_dialogue(file_name):
    dialogs = json.loads(open(file_name, encoding='utf-8-sig').read())
    items = []
    for dialog in dialogs:
        d_id=dialog["dialogue_id"]
        for turn in dialog["turns"]:
            for frame in turn["frames"]: 
                annotations = get_act_anntation_format(frame['actions'])
            speaker = turn["speaker"]
            uttr = turn["utterance"]
            an_item = dict(labels=annotations, speaker=speaker, uttr=uttr)
            items.append(an_item)
    return items, d_id

def rewrite_dialogue(template_file, uttrs, label_time):
    dialogs = json.loads(open(template_file, encoding='utf-8-sig').read())
    coherence_score = deepcopy(uttrs[0])
    for dialog in dialogs:
        for turn, par_uttr in zip(dialog["turns"], uttrs[1:]):
            turn["utterance"] = par_uttr[1]
            for frame in turn["frames"]:
                frame["slots"] = find_slot(par_uttr[1], frame["actions"])
    print(os.path.split(template_file)[1])
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    elements = [os.path.split(template_file)[1], str(label_time), coherence_score[1], dt_string]
    with open('./labeled_elements.txt', "a+") as ltf: ltf.write(','.join(elements)+'\n')
    with open(os.path.join('./labeled_dialog', os.path.split(template_file)[1]), 'w', encoding='utf-8') as f:
        json.dump(dialogs, f, ensure_ascii=False, indent=4)

@app.route('/', methods=['GET', 'POST'])
def index():
    dialogue_path = './new_json'
    dialogue_file = random.choice(os.listdir(dialogue_path))
    if request.method == 'POST':
       if not os.path.isdir('./labeled_dialog'): os.mkdir('./labeled_dialog')
       with open('./labeled_dialogue.txt', "r") as ldf:
        dialogue_file = [line.rstrip('\n') for line in ldf.readlines()]
        label_time = datetime.now().timestamp() - session['start_time']
       rewrite_dialogue(os.path.join(dialogue_path, dialogue_file[-1]), list(dict(request.form).items()), label_time)
       with open('./labeled_dialogue.txt', "w") as lf:
           for d_file in os.listdir('./labeled_dialog'):
            lf.write(d_file+'\n')    
       return redirect(url_for('index'))
       
    session['start_time'] = datetime.now().timestamp()
    with open('./labeled_dialogue.txt', "a+") as lf:
        labeled_files = [line.rstrip('\n') for line in lf.readlines()]
        while str(dialogue_file) in labeled_files:
            dialogue_file = random.choice(os.listdir(dialogue_path)) 
        lf.write(dialogue_file+'\n')        
    items, d_id = get_dialogue(os.path.join(dialogue_path, dialogue_file))
    return render_template('label_system.html', items=items, d_id=d_id)

@app.route('/history')
def history():
    items = []
    for label_dialog in os.listdir('labeled_dialog'):
        an_item = dict(d_id=label_dialog.replace('.json', ''))
        items.append(an_item)
    return render_template('history.html', items=items)

@app.route('/<file_name>')
def load_labeled_dialog(file_name):
    dialogue_path = './labeled_dialog'
    items, d_id = get_dialogue(os.path.join(dialogue_path, file_name+'.json'))
    return render_template('rewrited_dialogue.html', items=items, d_id=d_id)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3171, debug=True)
    #get_dialogue('./new_json/915.json')
    '''
    total_turns = 0
    for d_file in os.listdir('./new_json'):
        dialogs = json.loads(open(os.path.join('./new_json', d_file), encoding='utf-8-sig').read())
        for dialog in dialogs:    
            total_turns += len(dialog["turns"])
            
    print(total_turns/len(os.listdir('./new_json')))
    '''
    