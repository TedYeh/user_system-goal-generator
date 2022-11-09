import os, json

def get_action_times(paths):
    action_times = {}
    for path in paths:
        file_path = os.path.join('sgd', path)
        for d_file in os.listdir(file_path):
            file_name = os.path.join(file_path, d_file)
            json.loads(open(file_name, encoding='utf-8').read())
            

if __name__=="__main__":
    paths = ['train', 'test', 'dev']
    get_action_times(paths)