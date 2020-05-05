import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

def get_topic_diversity(beta):

    beta = softmax(beta, axis=1)
    logits = pairwise_distances(beta, metric='cosine')
    TD = logits[np.triu_indices(logits.shape[0], k = 1)].mean()
    print('Topic diveristy is: {}'.format(TD))

    return TD

def get_document_frequency(data, wi, wj=None):

    if wj is None:

        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if doc[wi]:
                D_wi += 1

        return D_wi

    D_wj = 0
    D_wi_wj = 0

    for l in range(len(data)):
        doc = data[l]

        if doc[wj]:
            D_wj += 1
            if doc[wi]:
                D_wi_wj += 1

    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):

    beta = softmax(beta, axis=1)
    D = len(data) ## number of docs...data is list of documents
    TC = []
    num_topics = len(beta)

    for k in range(num_topics):

        top_10 = list(beta[k].argsort()[-11:][::-1])
        TC_k = 0
        counter = 0

        for i, word in enumerate(top_10):

            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0

            while j < len(top_10) and j > i:
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                f_wi_wj = np.log(D_wi_wj + 1) - np.log(D_wi)

                #if D_wi_wj == 0:
                #    f_wi_wj = -1
                #else:
                #    f_wi_wj = -1 + ((np.log(D_wi_wj) - np.log(D)) - (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D))) / (-np.log(D_wi_wj) + np.log(D))

                tmp += f_wi_wj
                j += 1
                counter += 1

            TC_k += tmp 
        TC.append(TC_k)

    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))

    return TC
