import random

import numpy as np
import pandas as pd
import re
from transformers import BigBirdForMaskedLM, BigBirdTokenizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nltk
from nltk import pos_tag
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from transformers import MarianMTModel, MarianTokenizer
from transformers import logging

logging.set_verbosity_error()


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    clean_data = data.drop(
        columns=[
            "Unnamed: 0",
            "contrast",
            "goal",
            "goals2",
            "list",
            "metaphor",
            "moral",
            "question",
            "collective",
            "story",
            "date",
            "q9video",
            "sexd3",
            "sexd1",
            "sexd2",
            "charisma",
            "_merge",
            "prolific_score",
            "prolific_indicator_all",
            "text_length_all",
            "sex",
        ],
        inplace=False,
    )
    clean_data.dropna(inplace=True)
    return clean_data


def back_translation(text, source_language="english", target_language="french"):
    temperature = 1
    if isinstance(text, str):
        original_text = re.sub(r'\.{3}', '.', text)
    else:
        original_text = re.sub(r'\.{3}', '.', text.iloc[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
    back_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(device)
    back_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            
    with torch.no_grad():
        sentences = sent_tokenize(original_text)
        back_translated_sentences = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_ids = model.generate(**inputs, do_sample=False, temperature=temperature, max_length=512)
            intermediate_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            back_inputs = back_tokenizer(intermediate_text, return_tensors="pt", padding=True, truncation=True).to(device)
            back_translated_ids = back_model.generate(**back_inputs, do_sample=False, temperature=temperature,max_length=512)
            back_translated_text = back_tokenizer.decode(back_translated_ids[0], skip_special_tokens=True)
            back_translated_sentences.append(back_translated_text)

    final_text = " ".join(back_translated_sentences)
    return final_text
    



def noise_injection(text, char_insert_p=0.2, ocr_aug_p=0.1, word_swaping_aug_p=0.3):

    word_swaping_aug = naw.RandomWordAug(action="swap", aug_p=word_swaping_aug_p)
    text = word_swaping_aug.augment(text)

    char_aug = nac.RandomCharAug(action="insert", aug_char_p=char_insert_p)
    text = char_aug.augment(text)

    ocr_aug = nac.OcrAug(aug_char_p=ocr_aug_p)
    text = ocr_aug.augment(text)

    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def synonyme_replacement_tfidf_dropout(
    text, tfidf_scores, dropout_p=0.5, replace_p=0.5
):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    augmented_words = []

    for i, (word, pos) in enumerate(pos_tags):
        word_lower = word.lower()

        if word_lower in nltk.corpus.stopwords.words("english") or not word.isalpha():
            augmented_words.append(word)
            continue

        tfidf_score = tfidf_scores.get(word_lower, min(tfidf_scores.values()))
        max_tfidf = max(tfidf_scores.values())
        min_tfidf = min(tfidf_scores.values())
        normalized_tfidf = (tfidf_score - min_tfidf) / (max_tfidf - min_tfidf)

        dropout_prob = dropout_p * (1 - normalized_tfidf)
        replace_prob = replace_p * (1 - normalized_tfidf)

        p = random.random()

        if p < dropout_prob:
            continue

        elif p < dropout_prob + replace_prob:

            word_pos = get_wordnet_pos(pos)
            if word_pos is None:
                augmented_words.append(word)
                continue

            synonyms = []
            for syn in wordnet.synsets(word, pos=word_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != word_lower:
                        synonyms.append(synonym)

            if synonyms:
                syn_word = random.choice(synonyms)
                augmented_words.append(syn_word)

            else:
                augmented_words.append(word)

        else:
            augmented_words.append(word)

    augmented_text = " ".join(augmented_words)
    return augmented_text


def get_corpus(corpus_type="reuters"):
    if corpus_type == "reuters":
        sentences = reuters.sents()
        corpus = [" ".join(sentence) for sentence in sentences]
        return corpus

    if corpus_type == "brown":
        sentences = brown.sents()
        corpus = [" ".join(sentence) for sentence in sentences]
        return corpus


def compute_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    return tfidf_scores


def mask_by_tfidf_pos(text, tfidf_scores, mask_p = 0.3, target_pos=['NN', 'VB', 'JJ', 'RB']):
    if isinstance(text, str):
        text = re.sub(r'\.{3}', '.', text)
    else:
        text = re.sub(r'\.{3}', '.', text.iloc[0])
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    masked_tokens = words.copy()
    max_tfidf = max(tfidf_scores.values())
    min_tfidf = min(tfidf_scores.values())
    model_name = "google/bigbird-roberta-base"
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)

    for i, (word, pos) in enumerate(pos_tags):
        word_lower = word.lower()
        if not word.isalpha() or word_lower not in tfidf_scores:
            continue

        if not any(pos.startswith(target) for target in target_pos):
            continue

        tfidf = tfidf_scores[word_lower]
        normalized_tfidf = (tfidf - min_tfidf) / (max_tfidf - min_tfidf)
        mask_prob = mask_p * (1 - normalized_tfidf)
        p = random.random()

        if p < mask_prob:
            masked_tokens[i] = tokenizer.mask_token

    return " ".join(masked_tokens)


def contexual_bert_by_tfidf_pos(text, tfidf_scores, mask_p = 0.3, n_augments = 1, target_pos=['NN', 'VB', 'JJ', 'RB'], model_name = 'bert-base-uncased'):

    masked_text = mask_by_tfidf_pos(text, tfidf_scores, mask_p=0.3, target_pos=target_pos)
    model_name = "google/bigbird-roberta-base"
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)
    model = BigBirdForMaskedLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    inputs = tokenizer(masked_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_ids = logits[0, mask_token_index].argmax(dim=-1)
    
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    tokenized_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for idx, token in zip(mask_token_index.tolist(), predicted_tokens):
        tokenized_text[idx] = token 
        
    augmented_text = tokenizer.convert_tokens_to_string(tokenized_text[1:-1])
    return augmented_text


def data_augmentation(
    data,
    translation=False,
    source_language="english",
    target_language="french",
    noise=True,
    char_insert_p=0.2,
    ocr_aug_p=0.1,
    word_swaping_aug_p=0.2,
    dropout_p=0.3,
    replace_p=0.5,
    mask_p=0.5,
    n_augments=1,
    target_corpus="reuters",
):

    new_data = pd.DataFrame(columns=data.columns)
    corpus = get_corpus(corpus_type=target_corpus)
    tfidf_scores = compute_tfidf(corpus)

    for i in tqdm(range(data.shape[0])):
        text = data.iloc[i, :]["final_text"]
        for j in range(n_augments):
            p = random.random()
            new_text = ""
            if p < 0.3:
                new_text = text

                # if translation:
                new_text = back_translation(new_text, source_language, target_language)

                # new_text =synonyme_replacement_tfidf_dropout(new_text, tfidf_scores, dropout_p = dropout_p, replace_p = replace_p)
                # new_text = contexual_bert_by_tfidf_pos(new_text, tfidf_scores, mask_p = mask_p, n_augments = 1, target_pos=['NN', 'VB', 'JJ', 'RB'])[0]

                # if noise:
                new_text = noise_injection(
                    text=new_text,
                    char_insert_p=char_insert_p,
                    ocr_aug_p=ocr_aug_p,
                    word_swaping_aug_p=word_swaping_aug_p,
                )[0]

                new_data.loc[len(new_data)] = data.iloc[i].copy()
                new_data.loc[len(new_data) - 1, "final_text"] = new_text

            else:
                new_text = text

                new_text = synonyme_replacement_tfidf_dropout(
                    new_text, tfidf_scores, dropout_p=dropout_p, replace_p=replace_p
                )
                new_text = contexual_bert_by_tfidf_pos(
                    new_text,
                    tfidf_scores,
                    mask_p=mask_p,
                    n_augments=1,
                    target_pos=["NN", "VB", "JJ", "RB"],
                )

                # if noise:
                new_text = noise_injection(
                    text=new_text,
                    char_insert_p=char_insert_p,
                    ocr_aug_p=ocr_aug_p,
                    word_swaping_aug_p=word_swaping_aug_p,
                )[0]

                new_data.loc[len(new_data)] = data.iloc[i].copy()
                new_data.loc[len(new_data) - 1, "final_text"] = new_text

            new_data.loc[len(new_data) - 1, "participant_id"] = data.iloc[i][
                "participant_id"
            ]
            new_data.loc[len(new_data) - 1, "sentences"] = len(sent_tokenize(new_text))
            new_data.loc[len(new_data) - 1, "words"] = len(new_text.split())
            new_data.loc[len(new_data) - 1, "num_approvals"] = data.iloc[i][
                "num_approvals"
            ]
            new_data.loc[len(new_data) - 1, "num_rejections"] = data.iloc[i][
                "num_rejections"
            ]
            new_data.loc[len(new_data) - 1, "education"] = data.iloc[i]["education"]
            new_data.loc[len(new_data) - 1, "hones16"] = np.clip(
                data.iloc[i]["hones16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "emoti16"] = np.clip(
                data.iloc[i]["emoti16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "extra16"] = np.clip(
                data.iloc[i]["extra16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "agree16"] = np.clip(
                data.iloc[i]["agree16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "consc16"] = np.clip(
                data.iloc[i]["consc16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "openn16"] = np.clip(
                data.iloc[i]["openn16"] + np.random.normal(0, 0.1), 0, 5
            )
            new_data.loc[len(new_data) - 1, "time_taken"] = data.iloc[i][
                "time_taken"
            ] + np.random.normal(0, 2)
            new_data.loc[len(new_data) - 1, "age"] = data.iloc[i]["age"] + int(
                np.round(np.random.normal(0, 1))
            )
            new_data.loc[len(new_data) - 1, "gender"] = data.iloc[i]["gender"]
            new_data.loc[len(new_data) - 1, "ethnicity"] = data.iloc[i]["ethnicity"]
            new_data.loc[len(new_data) - 1, "employment"] = data.iloc[i]["employment"]
            new_data.loc[len(new_data) - 1, "status"] = data.iloc[i]["status"]
            new_data.loc[len(new_data) - 1, "incentive"] = data.iloc[i]["incentive"]
            new_data.loc[len(new_data) - 1, "icar"] = data.iloc[i][
                "icar"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "icar_hat0"] = data.iloc[i][
                "icar_hat0"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "icar_hat1"] = data.iloc[i][
                "icar_hat1"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "icar_hat2"] = data.iloc[i][
                "icar_hat2"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "overall_sentiment_all"] = data.iloc[i][
                "overall_sentiment_all"
            ]
            new_data.loc[len(new_data) - 1, "positive_sentiment_all"] = data.iloc[i][
                "positive_sentiment_all"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "negative_sentiment_all"] = data.iloc[i][
                "negative_sentiment_all"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "neutra_sentiment_all"] = data.iloc[i][
                "neutra_sentiment_all"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "mixed_sentiment_all"] = data.iloc[i][
                "mixed_sentiment_all"
            ] + np.random.normal(0, 0.1)
            new_data.loc[len(new_data) - 1, "targets"] = data.iloc[i]["targets"]

    dummy_new_data = pd.get_dummies(
        new_data,
        columns=[
            "education",
            "gender",
            "ethnicity",
            "employment",
            "status",
            "incentive",
            "overall_sentiment_all",
            "targets",
        ],
        drop_first=True,
        dtype=int,
    )
    return dummy_new_data
