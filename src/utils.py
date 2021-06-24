"""
This script contains helper functions for text cleaning, text processing and inference.

"""

import re
import torch
import numpy as np
from tqdm import tqdm

def is_ascii(w):
    try:
        w.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def decode_taxonomy(encoding):
    """
    Label decode taxonomy.

    Parameters:
    -----------
    encoding: int, taxonomy encoded

    Returns:
    --------
    decoding: int, taxonomy class
    """
    decoding_dict = {0: 2, 1: 9, 2: 59, 3: 70, 4: 173, 5: 268, 6: 273, 7: 280}
    decoding = decoding_dict[encoding]

    return decoding


def decode_subtaxonomy(subtaxonomy):
    """
    Label decode subtaxonomies of taxonomy 173.

    Parameters:
    -----------
    encoding: int, subtaxonomy encoded

    Returns:
    --------
    decoding: int, subtaxonomy class
    """
    decoding_dict = {0: 201, 1: 205, 2: 265, 3: 267, 4: 279}
    decoding = decoding_dict[subtaxonomy]

    return decoding


def text_cleaning(text):
    """
    Clean text from symbols, punctuation, etc.

    Parameters:
    -----------
    text: string, text data

    Returns:
    --------
    cleaned_text: string, cleaned text data
    """
    # remove string formatting '\n' or '\t'
    tmp_text = re.sub(r'\n+', '. ', text)
    tmp_text = re.sub(r'\t+', '. ', text)
    # remove words with non-ascii characters
    tmp_text = " ".join([word for word in tmp_text.split() if is_ascii(word)])
    # remove email address
    tmp_text = " ".join([word for word in tmp_text.split() if not word.startswith("@")])
    # remove urls
    tmp_text = re.sub(r'http\S+', '', tmp_text, flags=re.MULTILINE)
    tmp_text = re.sub(r'www\S+', '', tmp_text, flags=re.MULTILINE)
    # remove punctuation but . (to split sentences)
    cleaned_text = re.sub('[^A-Za-z.,]+', ' ', tmp_text)
    # lowercase
    cleaned_text = cleaned_text.lower()

    return cleaned_text


def text_preprocessing_a1(text):
    """
    Approach 1: Join first 2 sentences with last 2 sentences of the text.

    Parameters:
    -----------
    text: string, text data

    Returns:
    --------
    preprocessed_text: string, preprocessed text data
    """
    # sentence tokenize based on '. '
    sentences = text.split('. ')
    # get 2 first and 2 last sentences
    if len(sentences) >= 4:
        preprocessed_text = ". ".join(text.split('. ')[:2] + text.split('. ')[-2:])
        return preprocessed_text
    # if there are not 4 sentences, return full text
    else:
        preprocessed_text = text
        return preprocessed_text


def preprocessing_a1(df):
    """
    Cleaning and preprocessing following approach 1.

    Parameters:
    -----------
    df: Pandas DataFrame, df with `title`, and`description`

    Returns:
    --------
    preprocessed_df: Pandas DataFrame, df with `text` (preprocessed text)
    """
    # drop rows with missing descriptions and drop title
    cleaned_df = df.dropna(axis=0).drop('title', axis=1)
    # clean description
    cleaned_df['description'] = cleaned_df['description'].map(text_cleaning)
    # preprocess description
    preprocessed_df = cleaned_df.copy()
    preprocessed_df['description'] = cleaned_df['description'].map(text_preprocessing_a1)
    preprocessed_df = preprocessed_df.rename(columns={'description': 'text'})

    return preprocessed_df


def get_inference(data_loader, device, model):
    outputs_list = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            mask = d['mask']

            # send them to the cuda device we are using
            ids = ids.to(torch.device(device), dtype=torch.long)
            mask = mask.to(torch.device(device), dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask
            )
            outputs_list.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

    # get most probable class
    inferences = np.argmax(outputs_list, axis=1)

    return inferences
