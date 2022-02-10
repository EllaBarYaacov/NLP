from nltk.corpus import brown
import re

def divide_brown():
    all_brown = brown.tagged_sents(categories=["news"])
    n = len(all_brown)
    div = int(n/10)
    return all_brown[:n-div], all_brown[n-div:]

def get_training_set():
    return divide_brown()[0]

def get_test_set():
    return divide_brown()[1]

def get_known_words():
    words = set()
    sentences = get_training_set()
    for sentence in sentences:
        for word, tag in sentence:
            words.add(word)
    return words


def get_unknown_words():
    known = get_known_words()
    unknown = set()
    test_set = get_test_set()
    for sentence in test_set:
        for word, tag in sentence:
            if word not in known:
                unknown.add(word)
    return unknown

def get_low_freq_words(n):
    training_set = get_training_set()
    count_dict = dict()
    for sentence in training_set:
        for word, tag in sentence:
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
    return set(map(lambda x: x[0], filter(lambda x: x[1] < n, count_dict.items())))

def update_count_dict(key, value, count_dict):
    if key not in count_dict:
        count_dict[key] = dict()
    if value not in count_dict[key]:
        count_dict[key][value] = 1
    else:
        count_dict[key][value] += 1

def clean_tag(tag):
    return re.split('[\+-]', tag)[0]

def get_all_tags():
    all_sents = brown.tagged_sents(categories=["news"])
    all_tags = set()
    for sentence in all_sents:
        for word, tag in sentence:
            all_tags.add(clean_tag(tag))
    return all_tags