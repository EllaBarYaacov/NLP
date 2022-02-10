from brown import *

def create_POS_dict():
    pos_dict = dict()
    for sentence in get_training_set():
        for word, tag in sentence:
            update_count_dict(word, clean_tag(tag), pos_dict)
    return pos_dict

def create_MLE_dict():
    mle_dict = dict()
    for word, all_pos in create_POS_dict().items():
        mle_dict[word] = max(all_pos.keys(), key=lambda x: all_pos[x])
    return mle_dict

def get_MLE_error_rate():
    test_set = get_test_set()
    known = {True: 0, False: 0}
    unknown = {True: 0, False: 0}

    mle_dict = create_MLE_dict()

    for sentence in test_set:
        for word, tag in sentence:
            if word in mle_dict:
                known[clean_tag(tag) == mle_dict[word]] += 1
            else:
                unknown[tag == "NN"] += 1

    error_known = known[False] / float(known[True] + known[False])
    error_unknown = unknown[False] / float(unknown[True] + unknown[False])
    error_total = (known[False] + unknown[False]) / float(unknown[True] +
                                                          unknown[False] +
                                                          known[True] +
                                                          known[False])

    return error_known, error_unknown, error_total
