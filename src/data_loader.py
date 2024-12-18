import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from src.transformations import my_personality_target_normalization, idiap_target_normalization

def get_inputs_chunked_data(datapath_features, datapath_targets, features, datapath_features2 = None):

    if features == 'psycological':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        X_train_cleaned = X_train.drop(columns =['chunk_id', 'participant_id', 'contrast', 'goal', 'goals2',	'list', 'metaphor',	'moral',
                  'question','collective','story','date','q9video','sexd3','sexd1','sexd2',
                  'charisma','_merge','prolific_score','prolific_indicator_all',
                  'text_length_all', 'sex', 'icar_hat0', 'icar_hat1', 'icar_hat2', 'collective', 
                  'hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar', 'chunk_text'], inplace = False)
        X_train_cleaned = pd.get_dummies(X_train_cleaned, columns =['overall_sentiment_all', 'education','gender', 'ethnicity', 'employment', 'status', 'targets'] ,drop_first = True)
        bool_cols = X_train_cleaned.select_dtypes(include='bool').columns
        X_train_cleaned[bool_cols] = X_train_cleaned[bool_cols].astype(int)
        X_train_cleaned.columns = X_train_cleaned.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)

        return X_train_cleaned, y_train
    
    if features == 'embeddings':
        data = np.load(datapath_features)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)

        return cls_embeddings, y_train
    
    if features == 'combined':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        X_train_cleaned = X_train.drop(columns =['participant_id', 'contrast', 'goal', 'goals2',	'list', 'metaphor',	'moral',
                  'question','collective','story','date','q9video','sexd3','sexd1','sexd2',
                  'charisma','_merge','prolific_score','prolific_indicator_all',
                  'text_length_all', 'sex', 'icar_hat0', 'icar_hat1', 'icar_hat2', 'collective', 
                  'hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar', 'chunk_text'], inplace = False)
        X_train_cleaned = pd.get_dummies(X_train_cleaned, columns =['overall_sentiment_all', 'education','gender', 'ethnicity', 'employment', 'status', 'targets'] ,drop_first = True)
        bool_cols = X_train_cleaned.select_dtypes(include='bool').columns
        X_train_cleaned[bool_cols] = X_train_cleaned[bool_cols].astype(int)
        X_train_cleaned.columns = X_train_cleaned.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        data = np.load(datapath_features2)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        X = pd.concat([X_train_cleaned, cls_embeddings], axis = 1)
        X.columns = X.columns.astype(str)
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)
        
        return X, y_train
     
def get_inputs_data(datapath_features, datapath_targets, features, datapath_features2 = None):
    
    if features == 'psycological':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        X_train_cleaned = X_train.drop(columns =['participant_id', 'contrast', 'goal', 'goals2',	'list', 'metaphor',	'moral',
                  'question','collective','story','date','q9video','sexd3','sexd1','sexd2',
                  'charisma','_merge','prolific_score','prolific_indicator_all',
                  'text_length_all', 'sex', 'icar_hat0', 'icar_hat1', 'icar_hat2', 'collective', 
                  'hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar', 'final_text'], inplace = False)
        X_train_cleaned = pd.get_dummies(X_train_cleaned, columns =['overall_sentiment_all', 'education','gender', 'ethnicity', 'employment', 'status', 'targets'] ,drop_first = True)
        bool_cols = X_train_cleaned.select_dtypes(include='bool').columns
        X_train_cleaned[bool_cols] = X_train_cleaned[bool_cols].astype(int)
        X_train_cleaned.columns = X_train_cleaned.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)

        return X_train_cleaned, y_train
    

    if features == 'embeddings':
        data = np.load(datapath_features)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)

        return cls_embeddings, y_train
    
    if features == 'combined':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        X_train_cleaned = X_train.drop(columns =['participant_id', 'contrast', 'goal', 'goals2',	'list', 'metaphor',	'moral',
                  'question','collective','story','date','q9video','sexd3','sexd1','sexd2',
                  'charisma','_merge','prolific_score','prolific_indicator_all',
                  'text_length_all', 'sex', 'icar_hat0', 'icar_hat1', 'icar_hat2', 'collective', 
                  'hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar', 'final_text'], inplace = False)
        X_train_cleaned = pd.get_dummies(X_train_cleaned, columns =['overall_sentiment_all', 'education','gender', 'ethnicity', 'employment', 'status', 'targets'] ,drop_first = True)
        bool_cols = X_train_cleaned.select_dtypes(include='bool').columns
        X_train_cleaned[bool_cols] = X_train_cleaned[bool_cols].astype(int)
        X_train_cleaned.columns = X_train_cleaned.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        data = np.load(datapath_features2)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        X = pd.concat([X_train_cleaned, cls_embeddings], axis = 1)
        X.columns = X.columns.astype(str)
        y_train = pd.read_csv(datapath_targets)
        y_train.drop(columns= "Unnamed: 0", inplace = True)

        return X, y_train
    
def get_inputs_my_personality(datapath_features, datapath_targets, features, datapath_features2 = None):
    
    if features == 'psycological':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        y_train = pd.read_csv(datapath_targets)

        return X_train, y_train
    
    if features == 'embeddings':
        data = np.load(datapath_features)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        y_train = pd.read_csv(datapath_targets)

        return cls_embeddings, y_train
    
    if features == 'combined':
        data = pd.read_csv(datapath_features)
        X_train = data.iloc[:, 1:]
        data = np.load(datapath_features2)
        cls_embeddings = pd.DataFrame(data['cls_embeddings'])
        X_train = pd.concat([X_train, cls_embeddings], axis = 1)
        X_train.columns = X_train.columns.astype(str)
        y_train = pd.read_csv(datapath_targets)

        return X_train, y_train

def creating_my_personality_data(datapath):
    
    df_my_personality = pd.read_csv(datapath, encoding="ISO-8859-1")
    drop_list = ['#AUTHID', 'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN', 'DATE', 'NETWORKSIZE', 'BETWEENNESS',
             'NBETWEENNESS', 'DENSITY', 'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY']
    df_my_personality_cleaned = df_my_personality.drop(drop_list,axis=1)
    df_my_personality_cleaned.drop_duplicates(subset=['STATUS'], inplace=True)
    X_my_personality = df_my_personality_cleaned['STATUS']
    y_my_personality = df_my_personality_cleaned.drop(columns=['STATUS'],axis=1)
    X_my_personality =pd.DataFrame(X_my_personality)
    y_my_personality = my_personality_target_normalization(y_my_personality)

    return X_my_personality, y_my_personality

def creating_idiap_data(datapath):

    idiap_data = pd.read_excel(datapath)
    idiap_data = idiap_data.dropna()
    idiap_data.drop_duplicates(inplace=True)
    X_idiap = idiap_data['final_text']
    X_idiap = pd.DataFrame(X_idiap)
    y_idiap = idiap_data[['hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar_hat0', 'icar_hat1', 'icar_hat2']]
    y_idiap = idiap_target_normalization(y_idiap)

    return X_idiap, y_idiap, idiap_data

def creating_idiap_chunked_data(datapath):

    chunked_data = pd.read_csv(datapath)
    chunked_data = chunked_data.dropna()
    chunked_data.drop_duplicates(inplace=True)
    X_idiap_chunked = chunked_data[['chunk_text', 'Unnamed: 0']]
    X_idiap_chunked = pd.DataFrame(X_idiap_chunked)
    y_idiap_chunked = chunked_data[['hones16', 'emoti16', 'extra16', 'agree16', 'consc16', 'openn16', 'icar_hat0', 'icar_hat1', 'icar_hat2']]
    y_idiap_chunked = idiap_target_normalization(y_idiap_chunked)
    
    return X_idiap_chunked, y_idiap_chunked, chunked_data 

