from mle import *
from hmm import *


def a():
    return divide_brown()


def b_i():
    return {**create_MLE_dict(), **{word: 'NN' for word in get_unknown_words()}}


def b_ii():
    return get_MLE_error_rate()


def c_i():
    return get_transitions_table(), get_emmisions_table()


def c_iii():
    return test_HMM_error_rate(get_emmisions_table(), get_transitions_table())


def d_i():
    return get_emissions_table_add_one_smoothing()


def d_ii():
    return test_HMM_error_rate(get_emissions_table_add_one_smoothing(), get_transitions_table())


def e_ii():
    low_freq_map = map_low_freq_and_unknown(3)
    return test_HMM_error_rate(get_emissions_table_with_pseudowords(low_freq_map),
                               get_transitions_table(), low_freq_map)


def e_iii_error_rates():
    low_freq_map = map_low_freq_and_unknown(3)
    return test_HMM_error_rate(get_emissions_table_with_pseudowords_add_one_smoothing(low_freq_map),
                               get_transitions_table(), low_freq_map)

def e_iii_confusion_matrix():
    low_freq_map = map_low_freq_and_unknown(3)
    get_confusion_matrix(get_emissions_table_with_pseudowords_add_one_smoothing(low_freq_map),
                         get_transitions_table(), low_freq_map).to_csv("confusion_matrix.csv")