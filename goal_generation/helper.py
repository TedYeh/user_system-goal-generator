from ckiptagger import WS
import random 

ws = WS("./tagger")

def get_query_text(text):    
    ws_results = ws([text])[0]
    s = random.randint(0, len(ws_results)//2)
    e = random.randint(s, len(ws_results))
    return ''.join(ws_results[s:e+1])

# APIS
def FindMessage(message='', contact='', group=''):
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
