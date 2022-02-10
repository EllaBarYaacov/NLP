from brown import *
import re
import pandas as pd

def create_emmision_count_dict():
    tag_dict = dict()
    for sentence in get_training_set():
        for word, tag in sentence:
            update_count_dict(clean_tag(tag), word, tag_dict)
    return tag_dict

def create_transition_count_dict():
    transition_dict = dict()
    for sentence in get_training_set():
        n = len(sentence)
        tag_pairs = [("*", sentence[0][1])] + [(sentence[i][1], sentence[i+1][1]) for i in range(n - 1)] + [(sentence[n-1][1], "STOP")]
        for curr_tag, next_tag in tag_pairs:
            update_count_dict(clean_tag(curr_tag), clean_tag(next_tag), transition_dict)
    return transition_dict

def normalize_counting_dict(counting_dict):
    tag_dict = counting_dict
    emission_dict = dict()
    for tag, all_words in tag_dict.items():
        total_count = sum(all_words.values())
        emission_dict[tag] = {word: (all_words[word] / total_count) for word in all_words}
    return emission_dict

def get_emmisions_table():
    return normalize_counting_dict(create_emmision_count_dict())

def get_emissions_table_add_one_smoothing():
    count_dict = create_emmision_count_dict()
    all_words = get_unknown_words() | get_known_words()
    for tag, tag_dict in count_dict.items():
        for word in all_words:
            if word in tag_dict:
                tag_dict[word] += 1
            else:
                tag_dict[word] = 1
    return normalize_counting_dict(count_dict)


def get_transitions_table():
    return normalize_counting_dict(create_transition_count_dict())


def map_low_freq_and_unknown(n):
    return {x: get_category_for_word(x) for x in get_unknown_words() | get_low_freq_words(n)}

def get_emissions_count_dict_with_pseudowords(low_freq_map):
    training_set = get_training_set()
    count_dict = dict()
    for sentence in training_set:
        for i, (word, tag) in enumerate(sentence):
            new_word = low_freq_map[word] if word in low_freq_map else word
            update_count_dict(clean_tag(tag), new_word, count_dict)
    return count_dict

def get_emissions_table_with_pseudowords(low_freq_map):
    return normalize_counting_dict(get_emissions_count_dict_with_pseudowords(low_freq_map))

def get_emissions_table_with_pseudowords_add_one_smoothing(low_freq_map):
    count_dict = get_emissions_count_dict_with_pseudowords(low_freq_map)
    categories = {
        "_twoDigitsNumber", "_fourDigitsNumber", '_containsDigitAnd$', '_containsDigitAnd%',
        '_containsDigitAnd:', '_containsDigitAndDashOrSlash', '_otherNum',
        '_allCaps', '_initCap', '_otherCat'}
    all_words = get_known_words().difference(get_low_freq_words(3)) | categories
    for tag, tag_dict in count_dict.items():
        for word in all_words:
            if word in tag_dict:
                tag_dict[word] += 1
            else:
                tag_dict[word] = 1
    return normalize_counting_dict(count_dict)

def get_category_for_word(word):
    # numbers
    if any(char.isdigit() for char in word):
        if word.isdigit():
            if len(word) == 2:
                return "_twoDigitsNumber"
            elif len(word) == 4:
                return "_fourDigitsNumber"
        if word.find('$') != -1:
            return '_containsDigitAnd$'
        if word.find('%') != -1:
            return '_containsDigitAnd%'
        if word.find(':') != -1:
            return '_containsDigitAnd:'
        if word.find('-') != -1 or word.find('/') != -1:
            return '_containsDigitAndDashOrSlash'
        return '_otherNum'
    if re.sub("[^a-zA-Z]+", '', word).isupper():
        return '_allCaps'
    if word[0].isupper():
        return '_initCap'
    return '_otherCat'



class HMM:
    def __init__(self, emission_table, transitions_table, low_freq_words=None):
        self.e = emission_table
        self.q = transitions_table
        self.D = None
        self.low_freq_words = low_freq_words

    def get_e(self, word, tag):
        if self.low_freq_words and word in self.low_freq_words:
            return self.get_e(self.low_freq_words[word], tag)
        if tag not in self.e or word not in self.e[tag]:
            return 0.0
        return self.e[tag][word]

    def get_q(self, curr_tag, prev_tag):
        if prev_tag not in self.q or curr_tag not in self.q[prev_tag]:
            return 0.0
        return self.q[prev_tag][curr_tag]

    def _init_viterbi_table(self, n):
        self.D = {y: [(0.0, "") for _ in range(n)] for y in self.q}

    def _base_case(self, x1):
        for y in self.D:
            q = self.get_q(y, "*")
            e = self.get_e(x1, y)
            self.D[y][0] = (q * e, "*")

    def _step(self, k, xk):
        for yi in self.D:
            vals = []  # (0.5, NN), (0.2, DET) ....
            for yj in self.D:
                result = self.D[yj][k-1][0] * self.get_q(yi, yj)
                vals.append((result, yj))
            best = max(vals, key=lambda x: x[0])
            best_val, best_tag = best[0] * self.get_e(xk, yi), best[1]
            self.D[yi][k] = (best_val, best_tag)

    def _fill_viterbi_table(self, word_list):
        self._init_viterbi_table(len(word_list))
        self._base_case(word_list[0])
        for i in range(1, len(word_list)):
            self._step(i, word_list[i])

    def predict(self, word_list):
        n = len(word_list) - 1
        self._fill_viterbi_table(word_list)
        last_vals = [((self.D[yi][-1][0] * self.get_q("STOP", yi), self.D[yi][-1][1]), yi) for yi in self.D]
        best_last_val = max(last_vals, key=lambda x: (x[0][0]))
        curr_tuple, curr_column = best_last_val[0], n
        tag_list = [best_last_val[1]]
        while n > 0:
            tag_list.insert(0, curr_tuple[1])
            curr_tuple = self.D[curr_tuple[1]][n-1]
            n -= 1
        return tag_list

def test_HMM_error_rate(e, q, low_freq_map=None):
    hmm = HMM(e, q, low_freq_map)
    test_set = get_test_set()
    known_words = get_known_words()

    known = {True: 0, False: 0}
    unknown = {True: 0, False: 0}

    for sentence in test_set:
        predicted_tags = hmm.predict([x[0] for x in sentence])
        for i, word_tup in enumerate(sentence):
            if word_tup[0] in known_words:
                known[predicted_tags[i] == clean_tag(word_tup[1])] += 1
            else:
                unknown[predicted_tags[i] == clean_tag(word_tup[1])] += 1

    error_known = known[False] / float(known[True] + known[False])
    error_unknown = unknown[False] / float(unknown[True] + unknown[False])
    error_total = (known[False] + unknown[False]) / float(unknown[True] +
                                                          unknown[False] +
                                                          known[True] +
                                                          known[False])

    return error_known, error_unknown, error_total


def get_confusion_matrix(e, q, low_freq_map):
    hmm = HMM(e, q, low_freq_map)
    test_set = get_test_set()

    all_tags = sorted(get_all_tags())
    confusion_matrix = {tag1: {tag2: 0 for tag2 in all_tags} for tag1 in all_tags}

    for sentence in test_set:
        predicted_tags = hmm.predict([x[0] for x in sentence])
        for i, word_tup in enumerate(sentence):
            prediction = predicted_tags[i]
            answer = clean_tag(word_tup[1])
            confusion_matrix[answer][prediction] += 1
    return pd.DataFrame(confusion_matrix)
