import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    all_vecs = [word_to_vec[word] if word in word_to_vec else np.zeros((1,embedding_dim)) for word in sent.text]
    average_vec = sum(all_vecs) / len(all_vecs)
    as_torch = torch.from_numpy(average_vec).view(300)
    return as_torch.float()


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = torch.zeros(size)
    one_hot[ind] += 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    all_vecs = [get_one_hot(len(word_to_ind), word_to_ind[word]) for word in sent.text]
    ans = sum(all_vecs) / len(all_vecs)
    return ans


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {val: i for i, val in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """

    sent_vectors = np.zeros((seq_len, embedding_dim), dtype=np.float)
    for i, word in enumerate(sent.text):
        if i < seq_len and word in word_to_vec:
            sent_vectors[i] = word_to_vec[word]
    return sent_vectors


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=n_layers,
                                    dropout=dropout,
                                    bidirectional=True,
                                    batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim * 2 * n_layers, out_features=1)
        return

    def forward(self, text):
        out = self.lstm(text.float())
        concat_hn = torch.cat(out[1][0].unbind(0), 1)
        return self.linear(concat_hn)

    def predict(self, text):
        out = self.forward(text)
        sig = nn.Sigmoid()(out)
        return sig.round()


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=1)
        self.linear1.to(get_available_device())

    def forward(self, x):
        return self.linear1(x.to(get_available_device()))

    def predict(self, x):
        out = self.forward(x)
        sig = nn.Sigmoid()(out)
        return sig.round()


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    mistakes = (preds.to(get_available_device()) - y.to(get_available_device())).abs().sum()
    return 1 - (mistakes / len(y))


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    # reset grad
    optimizer.zero_grad()

    # do eval
    loss = torch.tensor([0.]).to(get_available_device())
    batch_accuracy_sum = 0
    batch_total = 0
    for batch_repr, batch_answers in data_iterator:
        batch_answers = batch_answers.view(len(batch_answers),1)
        batch_total += 1
        if batch_total > 20:
            break
        preds = model.forward(batch_repr.to(get_available_device()))
        loss += criterion.to(get_available_device())(preds.to(get_available_device()), batch_answers.to(get_available_device()))
        accuracy = binary_accuracy(model.predict(batch_repr.to(get_available_device())), batch_answers.to(get_available_device()))
        batch_accuracy_sum += accuracy
    assert batch_total != 0
    accuracy = batch_accuracy_sum / batch_total

    # do gradient step
    loss.backward()
    optimizer.step()

    return loss, accuracy


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """

    loss = torch.tensor([0.]).to(get_available_device())
    batch_accuracy_sum = 0
    batch_total = 0
    for batch_repr, batch_answers in data_iterator:
        batch_answers = batch_answers.view(len(batch_answers), 1)
        batch_total += 1
        preds = model.forward(batch_repr.to(get_available_device()))
        loss += criterion.to(get_available_device())(preds.to(get_available_device()), batch_answers.to(get_available_device()))
        accuracy = binary_accuracy(model.predict(batch_repr.to(get_available_device())), batch_answers.to(get_available_device()))
        batch_accuracy_sum += accuracy
        # if batch_total % 100 == 0:
        #     print(batch_total, accuracy)
    assert batch_total != 0
    accuracy = batch_accuracy_sum / batch_total

    return loss, accuracy


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = torch.empty(1, 1)
    for batch_repr, batch_answers in data_iter:
        predictions = torch.cat((predictions, model.predict(batch_repr)), 0)
    return predictions[1:]

def get_neg_pol_results():
    return

def get_rare_results():
    return

def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    # getting results for training data
    device = get_available_device()

    training_data_iterator = data_manager.get_torch_iterator(TRAIN)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    model.to(device)

    validation_data_iterator = data_manager.get_torch_iterator(VAL)

    loss_results = np.zeros((n_epochs,2))
    accuracy_results = np.zeros((n_epochs, 2))

    for i in range(n_epochs):
        t_loss, t_accuracy = train_epoch(model, training_data_iterator, optimizer, criterion)
        loss_results[i][0] = t_loss
        accuracy_results[i][0] = t_accuracy

        v_loss, v_accuracy = evaluate(model, validation_data_iterator, criterion)
        loss_results[i][1] = v_loss
        accuracy_results[i][1] = v_accuracy
        print("Epoch: {id}   Training loss: {tl}    Validation loss: {vl}".format(id=i, tl=t_loss, vl=v_loss))
        print("Epoch: {id}   Training accuracy: {tl}    Validation accuracy: {vl}".format(id=i, tl=t_accuracy, vl=v_accuracy))

    # getting results for test

    test_loss, test_accuracy = evaluate(model, data_manager.get_torch_iterator(TEST), criterion)
    print("Test loss: {} \n Test accuracy: {}".format(test_loss, test_accuracy))

    # # getting results for subsets
    dataset = data_manager.sentiment_dataset
    test_labels = data_manager.get_labels(data_subset=TEST)
    test_preds = get_predictions_for_data(model, data_manager.get_torch_iterator(data_subset=TEST))

    neg_indexes = data_loader.get_negated_polarity_examples(dataset.get_test_set())
    neg_pred_torch = test_preds[neg_indexes]
    neg_label_torch = torch.from_numpy(test_labels[neg_indexes])
    neg_accuracy = binary_accuracy(neg_pred_torch, neg_label_torch.view(neg_label_torch.shape[0], 1))
    neg_loss = criterion(test_preds[neg_indexes],
                         neg_label_torch.view(neg_label_torch.shape[0], 1))

    rare_indexes = data_loader.get_rare_words_examples(dataset.get_test_set(), dataset)
    rare_pred_torch = test_preds[rare_indexes]
    rare_label_torch = torch.from_numpy(test_labels[rare_indexes])
    rare_accuracy = binary_accuracy(rare_pred_torch, rare_label_torch.view(rare_label_torch.shape[0], 1))
    rare_loss = criterion(test_preds[rare_indexes],
                          rare_label_torch.view(rare_label_torch.shape[0], 1))

    print("Neg loss: {}\n"
          "Neg Accuracy: {}".format(neg_loss, neg_accuracy))
    print("Rare loss: {}\n"
          "Rare Accuracy: {}".format(rare_loss, rare_accuracy))
    return loss_results, accuracy_results, test_loss, test_accuracy, neg_loss, neg_accuracy, rare_loss, rare_accuracy



def train_log_linear_with_one_hot():

    # set model
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)
    input_size = data_manager.get_input_shape()[0]
    model = LogLinear(input_size)

    # hyper-parameters
    n_epochs = 20
    lr = 1e-1
    weight_decay = 0.0001

    return train_model(model, data_manager, n_epochs, lr, weight_decay)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    # set model
    print("loading data")
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    input_size = data_manager.get_input_shape()[0]
    model = LogLinear(input_size)

    # hyper-parameters
    n_epochs = 20
    lr = 1e-1
    weight_decay = 0.0001

    print("training model")
    return train_model(model, data_manager, n_epochs, lr, weight_decay)



def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """


    # set model
    print("Loading data")
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    input_size = data_manager.get_input_shape()[0]

    # hyper-parameters
    hidden_dim = 100
    dropout = 0.5
    n_layers = 1
    n_epochs = 4
    lr = 1e-2
    weight_decay = 0.0001

    model = LSTM(W2V_EMBEDDING_DIM, hidden_dim, dropout=dropout, n_layers=n_layers)

    print("Training model")
    return train_model(model, data_manager, n_epochs, lr, weight_decay)


if __name__ == '__main__':

    # results = train_log_linear_with_one_hot()
    # loss, accuracy = results

    # loss_res, acc_res, loss_test, acc_test = train_log_linear_with_w2v()


    # model = LSTM(embedding_dim=5, hidden_dim=4, n_layers=2, dropout=0)
    # x = torch.tensor(
    #     [[[1., 1., 1., 1., 1.],
    #       [0., 0., 0., 0., 1.]]])
    # out = model.predict(x)
    # print(out)

    # sent = ["hi", "hi", "hello", "moshe"]
    # word_to_vec = {"hi": np.array([1, 1, 1]), "hello": np.array([2, 1, 0])}
    #
    # ans = sentence_to_embedding(sent, word_to_vec, 6, 3)
    # print(ans)

    result = train_lstm_with_w2v()
    print(result)