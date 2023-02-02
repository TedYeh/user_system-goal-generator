from bs4 import BeautifulSoup
import requests, json, random
import os, time
from datetime import datetime
from pprint import pprint
from flask import Flask, flash, request, redirect, url_for, render_template, session
app = Flask(__name__)
app.config['SECRET_KEY'] = b'>\x89k\xff.t{\xed\xc0\x8c^E\x81A\xe7\xb6'

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
    for dialog in dialogs:
        for turn, par_uttr in zip(dialog["turns"], uttrs):
            turn["utterance"] = par_uttr[1]
    print(os.path.split(template_file)[1])
    with open('./labeled_time.txt', "a+") as ltf: ltf.write(os.path.split(template_file)[1]+' '+str(label_time)+'\n')
    with open(os.path.join('./labeled_dialog', os.path.split(template_file)[1]), 'w', encoding='utf-8') as f:
        json.dump(dialogs, f, ensure_ascii=False, indent=4)

@app.route('/', methods=['GET', 'POST'])
def index():
    dialogue_path = './new_json'
    dialogue_file = random.choice(os.listdir(dialogue_path))
    if request.method == 'POST':
       if not os.path.isdir('./labeled_dialog'): os.mkdir('./labeled_dialog')
       with open('./labeled_dialogue.txt', "r") as lf:
        dialogue_file = [line.rstrip('\n') for line in lf.readlines()]
        label_time = datetime.now().timestamp() - session['start_time']
       rewrite_dialogue(os.path.join(dialogue_path, dialogue_file[-1]), dict(request.form).items(), label_time)
       return redirect(url_for('index'))
       
    session['start_time'] = datetime.now().timestamp()
    with open('./labeled_dialogue.txt', "a+") as lf:
        labeled_files = [line.rstrip('\n') for line in lf.readlines()]
        while str(dialogue_file) in labeled_files:
            dialogue_file = random.choice(os.listdir(dialogue_path)) 
        lf.write(dialogue_file+'\n')        
    items, d_id = get_dialogue(os.path.join(dialogue_path, dialogue_file))
    return render_template('label_system.html', items=items, d_id=d_id)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    #get_dialogue('./new_json/915.json')
    