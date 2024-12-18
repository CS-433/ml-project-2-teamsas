import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

def idiap_target_normalization(y_idiap):

    for col in ['hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar_hat0', 'icar_hat1', 'icar_hat2']:
        y_idiap[col] = y_idiap[col] - y_idiap[col].min()
        y_idiap[col] = y_idiap[col] / y_idiap[col].max()

    return y_idiap

def my_personality_target_normalization(y_my_personality):

    for col in ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']:
        y_my_personality[col] = y_my_personality[col] - y_my_personality[col].min()
        y_my_personality[col] = y_my_personality[col] / y_my_personality[col].max()

    return y_my_personality

def normalization_readability(df):
    
    for idx, col in enumerate(df.columns):
        df[col] = np.nan_to_num(stats.zscore(df[col]))

    return df
