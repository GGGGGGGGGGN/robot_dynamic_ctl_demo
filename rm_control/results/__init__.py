import os   

def get_result_dir():
    return os.path.dirname(__file__)

def get_result_fig_dir():
    return os.path.join(os.path.dirname(__file__), "figures")