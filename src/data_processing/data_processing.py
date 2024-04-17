import numpy as np
import pandas as pd
import re
from collections import Counter
import json


# Standardizing salaries to an annual average
def standardize_salary(row):
    if pd.isnull(row['med_salary']):
        avg_salary = np.mean([row['max_salary'], row['min_salary']])
    else:
        avg_salary = row['med_salary']

    if row['pay_period'] == 'MONTHLY':
        return avg_salary * 12
    elif row['pay_period'] == 'WEEKLY':
        # Assuming a full-time job works 48 weeks a year
        return avg_salary * 48
    elif row['pay_period'] == 'HOURLY':
        # Assuming a full-time job works 1920 hours a year (40 hours/week * 48 weeks)
        return avg_salary * 1920
    else:  # Assuming the salary is already yearly
        return avg_salary


# Clean the descriptions
def clean_description(description, tokenizer):
    description = description.lower()
    description = re.sub(r'[^a-zA-Z0-9\s]', ' ', description)
    doc = tokenizer(description)
    tokens = [token.text for token in doc]
    return ' '.join(tokens)


vocab_size = 400


def clean_title(title, tokenizer):
    # Lowercase
    title = title.lower()
    # Remove special characters and numbers
    title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
    # Remove multiple spaces
    title = re.sub('\\s+', ' ', title)
    title = ' '.join([w.text for w in tokenizer(title) if not w.is_stop])
    return title


def generate_vocabulary(cleaned_titles):
    tokens = []
    for title in cleaned_titles:
        tokens.extend(title.split())
    word_freq = Counter(tokens)
    vocab = sorted(word_freq, key=word_freq.get, reverse=True)[:vocab_size]
    with open('title_vocabulary.json', 'w') as f:
        json.dump(vocab, f)
    return vocab


def create_title_emb(cleaned_title, word_to_vec):
    tokens = cleaned_title.split()
    a = [[0.0 for _ in range(vocab_size)]]
    b = [word_to_vec[word] for word in tokens if word in word_to_vec]
    c = a + b
    embeddings = np.array(c)
    embedding = np.sum(embeddings, axis=0)
    return embedding

#
#
# def clean_title(title):
#     # Lowercase
#     title = title.lower()
#     # Remove special characters and numbers
#     title = re.sub('[^A-Za-z ]+', ' ', title)
#     # Remove multiple spaces
#     title = re.sub('\\s+', ' ', title)
#     return title
#
#
# def generate_vocabulary(data, tokenizer):
#     titles = data.title.to_numpy()
#     # Clean data
#     cleaned_titles = list(map(clean_title, titles))
#
#     tokens = []
#     for title in cleaned_titles:
#         tokens.extend([w.text for w in tokenizer(title) if not w.is_stop])
#     word_freq = Counter(tokens)
#     vocab = sorted(word_freq, key=word_freq.get, reverse=True)[:200]
#     with open('title_vocabulary.json', 'w') as f:
#         json.dump(vocab, f)
#     return vocab, cleaned_titles
#
#
# def tokenize_title(title, word_to_vec, tokenizer):
#     cleaned_title = clean_title(title)
#     tokens = [w.text for w in tokenizer(cleaned_title)]
#     embeddings = [word_to_vec[word] for word in tokens if word in word_to_vec]
#     return embeddings
#
#
# if __name__ == '__main__':
#     nlp = spacy.load("en_core_web_sm")
#     data = pd.read_csv("./raw_data/job_postings.csv")
#     # vocab, cleaned_titles = generate_vocabulary(data)
#     with open('title_vocabulary.json', 'r') as f:
#         vocab = json.load(f)
#
#     word_to_vec = {word: np.eye(len(vocab))[i] for i, word in enumerate(vocab)}
#
#     for title in data.title.to_numpy():
#         emb = tokenize_title(title, word_to_vec, nlp)
#         print(emb)
