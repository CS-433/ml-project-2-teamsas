import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import readability
from empath import Empath
from src.transformations import normalization_readability 

def get_wordnet_pos(word):
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def extract_NRC_features(x, NRC_df):
    
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(x)
    processed_tokens = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) for word in tokens if word.lower() not in stop_words and word not in punctuation]
    tokens = Counter(processed_tokens)
    df = pd.DataFrame.from_dict(tokens, orient="index", columns=["count"])
    merged_df = pd.merge(df, NRC_df, left_index=True, right_index=True)

    if merged_df.empty:
        new_row = {'Positive': 0, 'Negative' : 0, 'Anger': 0, 'Anticipation': 0, 'Disgust': 0, 'Fear': 0, 'Joy': 0, 'Sadness': 0, 'Surprise': 0, 'Trust': 0, 'count': 0}
        result = pd.DataFrame( new_row, index=[0]).iloc[0]

        return result
    
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]

    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]

    return result

def extract_NRC_VAD_features(x, NRC_df):
    
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(x)
    processed_tokens = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) for word in tokens if word.lower() not in stop_words and word not in punctuation]
    tokens = Counter(processed_tokens)
    df = pd.DataFrame.from_dict(tokens, orient="index", columns=["count"])
    merged_df = pd.merge(df, NRC_df, left_index=True, right_index=True)

    if merged_df.empty:
        new_row = {'Valence': 0, 'Arousal' : 0, 'Dominance': 0}
        result = pd.DataFrame( new_row, index=[0]).iloc[0]

        return result

    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]

    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]

    return result

def extract_readability_features(text):
    
    try:
        features = dict(readability.getmeasures(text, lang="en"))
        result = {}
        for d in features:
            if d == "word usage" or d == "sentence beginnings":
                continue
            result.update(features[d])
        del result["paragraphs"]
        del result["characters_per_word"]
        del result["sentences"]
        del result["words"]
        del result["syllables"]
        del result["characters"]
        del result["wordtypes"]
        del result["sentences_per_paragraph"]
        del result["words_per_sentence"]
        del result["syll_per_word"]
        return pd.Series(result) 
            
    except Exception as e:
        new_row = {'Kincaid': 0, 'ARI': 0, 'Coleman-Liau': 0, 'FleschReadingEase': 0, 'GunningFogIndex': 0, 'LIX': 0, 'SMOGIndex': 0, 'RIX': 0, 'DaleChallIndex': 0, 'type_token_ratio':0, 'long_words':0, 'complex_words':0, 'complex_words_dc':0}
        
        return pd.Series(new_row)

def exraxt_more_emotion_features(text):

    lexicon = Empath()
    analysis = lexicon.analyze(text, normalize=True)
    desired_keys = ["warmth","emotional","sympathy","love","confusion", "cheerfulness", "aggression", "nervousness", "ridicule", "royalty", "optimism"]
    subset = {key: analysis[key] for key in desired_keys if key in analysis}

    return pd.Series(subset)

def saving_NRC_data(NRC_path, X, output_path, type):

    NRC_df = pd.read_excel(NRC_path, index_col=0)    
    if type == 'my_personality':
        features = X["STATUS"].apply(lambda x: extract_NRC_features(x, NRC_df))
    if type == 'idiap':
        features = X["final_text"].apply(lambda x: extract_NRC_features(x, NRC_df))
    if type == 'idiap_chunked':
        features = X["chunk_text"].apply(lambda x: extract_NRC_features(x, NRC_df))
    result = pd.concat([X, features], axis=1)
    if type != 'idiap_chunked':
        result.drop(columns=["count"], inplace=True)
    result.to_csv(output_path, index=False)


def saving_NRC_VAD_data(NRC_path, X, output_path, type):

    NRC_df = pd.read_csv(NRC_path, index_col=["Word"], sep="\t")
    if type == 'my_personality':
        features = X["STATUS"].apply(lambda x: extract_NRC_VAD_features(x, NRC_df))
    if type == 'idiap':
        features = X["final_text"].apply(lambda x: extract_NRC_VAD_features(x, NRC_df))
    if type == 'idiap_chunked':
        features = X["chunk_text"].apply(lambda x: extract_NRC_VAD_features(x, NRC_df))
    result = pd.concat([X, features], axis=1)
    result.to_csv(output_path, index=False)


def saving_readability_data(X, output_path, type):

    if type == 'my_personality':
        features = X["STATUS"].apply(lambda x: extract_readability_features(x))
    if type == 'idiap':
        features = X["final_text"].apply(lambda x: extract_readability_features(x))
    if type == 'idiap_chunked':
        features = X["chunk_text"].apply(lambda x: extract_readability_features(x))
    features = normalization_readability(features)
    result_readability = pd.concat([X, features], axis=1)
    result_readability.to_csv(output_path, index=False)


def saving_LIWC_data(X, output_path, type):

    if type == 'my_personality':
        features = X["STATUS"].apply(lambda x: exraxt_more_emotion_features(x))
    if type == 'idiap':
        features = X["final_text"].apply(lambda x: exraxt_more_emotion_features(x))
    if type == 'idiap_chunked':
        features = X["chunk_text"].apply(lambda x: exraxt_more_emotion_features(x))
    result_emotions = pd.concat([X, features], axis=1)
    result_emotions.to_csv(output_path, index=False)


def Saving_aggregated_features(X, y, type, Path_NRC, Path_NRC_VAD, Path_readability, Path_LIWC, output_path, target_path, data = None):
    
    df_NRC_features = pd.read_csv(Path_NRC)
    df_NRC_vad_features = pd.read_csv(Path_NRC_VAD)
    df_readability_features = pd.read_csv(Path_readability)
    df_emotion_features = pd.read_csv(Path_LIWC)
    if type == 'my_personality':
        ON = "STATUS"
    if type == 'idiap':
        ON = "final_text"
    if type == 'idiap_chunked':
        ON = "Unnamed: 0"

    df_merged_1 = pd.merge(X, df_NRC_features, how="left", on = ON)
    if type == 'idiap_chunked':
        df_merged_1.drop(columns=["chunk_text_y"], inplace=True)
        df_merged_1.rename(columns={"chunk_text_x": "chunk_text"}, inplace=True)

    df_merged_2 = pd.merge(df_merged_1, df_NRC_vad_features, how="left", on=ON)
    if type == 'idiap_chunked':
        df_merged_2.drop(columns=["chunk_text_y"], inplace=True)
        df_merged_2.rename(columns={"chunk_text_x": "chunk_text"}, inplace=True)

    df_merged_3 = pd.merge(df_merged_2, df_readability_features, how="left", on=ON)
    if type == 'idiap_chunked':
        df_merged_3.drop(columns=["chunk_text_y"], inplace=True)
        df_merged_3.rename(columns={"chunk_text_x": "chunk_text"}, inplace=True)

    df_merged_4 = pd.merge(df_merged_3, df_emotion_features, how="left", on=ON)
    if type == 'idiap_chunked':
        df_merged_4.drop(columns=["chunk_text_y"], inplace=True)
        df_merged_4.rename(columns={"chunk_text_x": "chunk_text"}, inplace=True)

    if type == "my_personality":
        df_merged_4.to_csv(output_path, index=False)

    if type == "idiap" or type == "idiap_chunked":
        df_all_features = pd.merge(data, df_merged_4, how="left", on=ON)
        if type == 'idiap_chunked':
            df_all_features.drop(columns=["chunk_text_y"], inplace=True)
            df_all_features.rename(columns={"chunk_text_x": "chunk_text"}, inplace=True)
        df_all_features.to_csv(output_path, index=False)
        
    y.to_csv(target_path, index=False)


