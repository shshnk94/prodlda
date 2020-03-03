import numpy as np

def get_topic_diversity(beta, topk):

    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))

    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx

    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
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

                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))

                tmp += f_wi_wj
                j += 1
                counter += 1

            TC_k += tmp 
        TC.append(TC_k)

    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))

    return TC
