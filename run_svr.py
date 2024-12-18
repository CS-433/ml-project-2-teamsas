from data_loader import get_inputs_my_personality, get_inputs_data, get_inputs_chunked_data
from learning import Regression_SVR

def run_svr(type, datapath_features, datapath_targets, target, features, datapath_features2 = None):
    if type == 'my_personality':
        X_my_personality, y_my_personality = get_inputs_my_personality(datapath_features = datapath_features, datapath_targets = datapath_targets, features = features, datapath_features2 = datapath_features2)
        Regression_SVR(X = X_my_personality, y = y_my_personality, type = 'my_personality',  target = target, features = features)
        
    elif type == 'idiap':
        X_idiap, y_idiap = get_inputs_data(datapath_features = datapath_features, datapath_targets = datapath_targets, features = features, datapath_features2 = datapath_features2)
        Regression_SVR(X = X_idiap, y = y_idiap, type = 'idiap',  target = target, features = features)    
    
    
    elif type == 'idiap_chunked':
        X_idiap, y_idiap = get_inputs_chunked_data(datapath_features = datapath_features, datapath_targets = datapath_targets, features = features, datapath_features2 = datapath_features2)
        Regression_SVR(X = X_idiap, y = y_idiap, type = 'idiap_chunked',  target = target, features = features)