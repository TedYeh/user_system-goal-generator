import random, re, json

def get_query_text(text):  
    from ckiptagger import WS  
    ws = WS("./tagger")
    ws_results = ws([text])[0]
    s = random.randint(0, len(ws_results)//2)
    e = random.randint(s, len(ws_results))
    return ''.join(ws_results[s:e+1])

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

def find_slot(uttrance, acts):
    slots = []    
    for a in acts:        
        act, _, slot, value = list(a.values())
        if act == "INFORM": 
            if value[0]:
                for match in re.finditer(re.escape(value[0]), uttrance):
                    slots.append({"exclusive_end": match.end(), "slot": slot, "start": match.start()})
    return slots

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

# APIS
def FindMessage(message, contact='', group=''):
    return [
              "contact_name",
              "group_name",
              "message"
            ]

def SendMessage(message, contact='', group=''):
    return [
              "contact_name",
              "group_name", 
              "message"
            ]

def FindMail(subject='', recipient='', sender='', content='', copy_recipient=''):
    return [
              "sender",
              "recipient",
              "subject",
              "content",
              "copy_recipient"
            ]

def SendMail(subject, recipient, content, copy_recipient=''):
    return [
              "recipient",
              "copy_recipient",
              "subject",
              "content"
            ]

def LookupEvents(event_name, event_time='', event_date='', event_content='', event_location='', participant=''):
    return [
                "event_date",
                "event_time",
                "event_location",
                "event_name"
            ]

def AddEvent(event_name, event_time, event_date, event_content='', event_location='', participant=''):
    return [
                "event_date",
                "event_time",
                "event_name",
                "event_content",
                "event_location",
                "participant"
            ]
