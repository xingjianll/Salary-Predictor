import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
import re
from collections import Counter
import json



def clean_title(title):
    # Lowercase
    title = title.lower()
    # Remove special characters and numbers
    title = re.sub('[^A-Za-z ]+', ' ', title)
    # Remove multiple spaces
    title = re.sub('\\s+', ' ', title)
    return title


def generate_vocabulary(data, tokenizer):
    titles = data.title.to_numpy()
    # Clean data
    cleaned_titles = list(map(clean_title, titles))

    tokens = []
    for title in cleaned_titles:
        tokens.extend([w.text for w in tokenizer(title) if not w.is_stop])
    word_freq = Counter(tokens)
    vocab = sorted(word_freq, key=word_freq.get, reverse=True)[:200]
    with open('title_vocabulary.json', 'w') as f:
        json.dump(vocab, f)
    return vocab, cleaned_titles


def tokenize_title(title, word_to_vec, tokenizer):
    cleaned_title = clean_title(title)
    tokens = [w.text for w in tokenizer(cleaned_title)]
    embeddings = [word_to_vec[word] for word in tokens if word in word_to_vec]
    return embeddings



if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    data = pd.read_csv("./raw_data/job_postings.csv")
    # vocab, cleaned_titles = generate_vocabulary(data)
    with open('title_vocabulary.json', 'r') as f:
        vocab = json.load(f)

    word_to_vec = {word: np.eye(len(vocab))[i] for i, word in enumerate(vocab)}

    for title in data.title.to_numpy():
        emb = tokenize_title(title, word_to_vec, nlp)
        print(emb)

