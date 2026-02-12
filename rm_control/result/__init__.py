import os   

def get_result_path():
    return os.path.dirname(__file__)

def get_result_fig_path():
    return os.path.join(os.path.dirname(__file__), "figures")