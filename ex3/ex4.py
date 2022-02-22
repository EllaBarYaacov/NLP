import numpy as np
import pandas as pd
from scipy import spatial
import scipy
import re
#
# def cosine_similarity(a, b):
#     return np.dot(a,b)/ (np.linalg.norm(a) * np.linalg.norm(b))
#     # 1 - spatial.distance.cosine(a, b)


sentences = ['John likes NLP', 'He likes Mary', 'John likes machine learning',
             'Deep learning is a subfield of machine learning', 'John wrote a post about NLP and got likes']


sentences = [re.sub("\\b(is|a|of|and)\\b", "", x ).replace("  ", " ").replace("  ", " ") for x in sentences]

all_words = []
for sentence in sentences:
    for word in sentence.split(' '):
        if word not in ['is', 'a', 'of', 'and']:
            all_words.append(word)
unique_words = list(set(all_words))

df = pd.DataFrame(np.zeros((len(unique_words), len(unique_words))).astype(int), columns=unique_words,index=unique_words)

for sentence in sentences:
    new_sentence = sentence.split((' '))
    for i, word in enumerate(new_sentence):
        if i == len(new_sentence) - 1:
            continue
        next_word = new_sentence[i + 1]
        df[word][next_word] += 1
        df[next_word][word] += 1



u, s, vt = np.linalg.svd(df.values)



# pd.DataFrame(u).to_csv("U.csv")
# pd.DataFrame(vt).to_csv("vt.csv")
# pd.DataFrame(s).to_csv("s.csv")


# s, u = scipy.linalg.eigh(df.values)
# vt = u.transpose()
reduced_size = int(0.3 * len(s))

reduced_u = pd.DataFrame(u[:, :reduced_size], index=unique_words)
reduced_s = s[:reduced_size]
reduced_vt = pd.DataFrame(vt[:reduced_size], columns=unique_words)

John_He =  1 - spatial.distance.cosine(reduced_u.loc['John'], reduced_u.loc['He'])
John_subfield = 1 - spatial.distance.cosine(reduced_u.loc['John'], reduced_u.loc['subfield'])
Deep_machine = 1 - spatial.distance.cosine(reduced_u.loc['Deep'], reduced_u.loc['machine'])
wrote_post = 1 - spatial.distance.cosine(reduced_u.loc['wrote'], reduced_u.loc['post'])


#
# print(John_He)
# print(John_subfield)
# print(Deep_machine)
print(wrote_post)


