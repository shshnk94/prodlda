import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
#import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda, nvlda
from scipy.special import softmax

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
#from ray.tune.suggest import ConcurrencyLimiter

from os import path
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

np.random.seed(0)
tf.set_random_seed(0)

m = ''
f = ''
s = ''
t = ''
b = ''
r = ''
e = ''
fold = ''

try:
    opts, args = getopt.getopt(sys.argv[1:],"hpnm:f:s:t:b:r:e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","learning_rate=","training_epochs","data_path=","save_path=","fold=", "mode=", "load_from="])
except getopt.GetoptError:
    print('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>')
        sys.exit()
    elif opt == '-p':
        print('Running with the Default settings for prodLDA...')
        print('CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100')
        m='prodlda'
        f=100
        s=100
        t=50
        b=200
        r=0.002
        e=100
    elif opt == '-n':
        print('Running with the Default settings for NVLDA...')
        print('CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300')
        m='nvlda'
        f=100
        s=100
        t=50
        b=200
        r=0.01
        e=300
    elif opt == "--mode":
        mode = arg
    elif opt == "-m":
        m=arg
    elif opt == "-f":
        f=int(arg)
    elif opt == "-s":
        s=int(arg)
    elif opt == "-t":
        t=int(arg)
    elif opt == "-b":
        b=int(arg)
    elif opt == "-r":
        r=float(arg)
    elif opt == "-e":
        e=int(arg)
    elif opt == "--data_path":
        data_path = arg
    elif opt == "--save_path":
        save_path = arg
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
    elif opt == "--load_from":
        load_from = arg
    elif opt == "--fold":
        fold = arg

#ckpt = os.path.join(save_path, 'k{}_e{}_lr{}'.format(t, e, r), 'fold{}'.format(fold)) if fold != '' else save_path

'''-----------Data--------------'''
def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

#fold_data_path = os.path.join(data_path, 'fold{}'.format(fold)) if fold != '' else data_path

def load_data(mode):

    vocab = pickle.load(open(os.path.join(data_path, 'vocab.pkl'),'rb'))
    
    def read(name):

        data = np.load(os.path.join(data_path, name), encoding='latin1', allow_pickle=True)
        data = np.array([onehot(doc.astype('int'), len(vocab)) for doc in data if np.sum(doc)!=0])
        
        return data
 
    if mode == 'train':

        data = read('train.txt.npy')
        return vocab, data
    
    else:

        if mode == 'valid':
            data_h1 = read('valid_h1.txt.npy')
            data_h2 = read('valid_h2.txt.npy')
        
        else:
            data_h1 = read('test_h1.txt.npy')
            data_h2 = read('test_h2.txt.npy')

        return vocab, data_h1, data_h2

#data_tr = np.load(fold_data_path + '/train.txt.npy', encoding='latin1', allow_pickle=True)

#if mode == 'train':
#    data_val_h1 = np.load(fold_data_path + '/valid_h1.txt.npy', encoding='latin1', allow_pickle=True)
#    data_val_h2 = np.load(fold_data_path + '/valid_h2.txt.npy', encoding='latin1', allow_pickle=True)

#data_te_h1 = np.load(data_path + '/test_h1.txt.npy', encoding='latin1', allow_pickle=True)
#data_te_h2 = np.load(data_path + '/test_h2.txt.npy', encoding='latin1', allow_pickle=True)

#vocab = data_path + '/vocab.pkl'
#vocab = pickle.load(open(vocab,'rb'))
#vocab_size=len(vocab)

#--------------convert to one-hot representation------------------
#print('Converting data to one-hot representation')
#data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])

#if mode == 'train':
#    data_val_h1 = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_val_h1 if np.sum(doc)!=0])
#    data_val_h2 = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_val_h2 if np.sum(doc)!=0])

#data_te_h1 = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_te_h1 if np.sum(doc)!=0])
#data_te_h2 = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_te_h2 if np.sum(doc)!=0])

#--------------print the data dimentions--------------------------
#print('Data Loaded')
#print('Dim Training Data',data_tr.shape)
#if mode == 'train':
#    print('Dim Validation Data',data_val_h1.shape)
#print('Dim Test Data',data_te_h1.shape)
'''-----------------------------'''

'''--------------Global Params---------------'''
#n_samples_tr = data_tr.shape[0]

#if mode == 'train':
#    n_samples_val = data_val_h1.shape[0]

#n_samples_te = data_te_h1.shape[0]

#docs_tr = data_tr
#docs_te = data_te
#docs_te_h1 = data_te_h1
#docs_te_h2 = data_te_h2

#if mode == 'train':
#    docs_val_h1 = data_val_h1
#    docs_val_h2 = data_val_h2


#batch_size=int(b)
#learning_rate=float(r)
"""
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
         n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space
"""
'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''

def make_network(layer1=100, layer2=100, num_topics=50, bs=200, eta=0.002):

    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate

'''--------------Methods--------------'''
def create_minibatch(data):
    #rng = np.random.RandomState(10)
    rng = np.random.RandomState(0)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

def get_summaries(sess):

  weights = tf.trainable_variables()
  values = [sess.graph.get_tensor_by_name(w.name) for w in weights]
  #values = weights.eval(sess)

  weight_summaries = []
  for weight, value in zip(weights, values):
    weight_summaries.append(tf.summary.histogram(weight.name, value))

  summaries = tf.summary.merge(weight_summaries) 
  #epoch = tf.placeholder(tf.float32, name="epoch")
  #writer.add_summary(, epoch)

  return summaries
 

#def train(network_architecture, minibatches, type='prodlda',learning_rate=0.001,
#          batch_size=200, training_epochs=100, display_step=5):
def train(config):

    tf.reset_default_graph()

    vocab, docs_tr = load_data('train')
    vocab, docs_va_h1, docs_va_h2 = load_data('valid')

    #if type=='prodlda':
        #vae = prodlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    network_architecture = {'n_hidden_recog_1': config['layer1'],
                            'n_hidden_recog_2': config['layer2'],
                            'n_hidden_gener_1': docs_tr.shape[1],
                            'n_input': docs_tr.shape[1],
                            'n_z': int(t)}
    vae = prodlda.VAE(network_architecture, learning_rate=config['learning_rate'], batch_size=config['batch_size'], keep_prob=config['keep_prob'])
    saver = tf.train.Saver()
    vae.sess.graph.finalize()

    #elif type=='nvlda':
    #    vae = nvlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    #summaries = get_summaries(vae.sess)
    #writer = tf.summary.FileWriter(ckpt + '/logs/', vae.sess.graph)

    for epoch in range(config['epochs']):

        avg_cost = 0.
        #total_batch = int(n_samples_tr / batch_size)
        indices = np.random.permutation(docs_tr.shape[0])
        for base in range(0, docs_tr.shape[0], config['batch_size']):
            batch_xs = docs_tr[indices[base: min(base + config['batch_size'], docs_tr.shape[0])]]
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / docs_tr.shape[0] * config['batch_size']

            if np.isnan(avg_cost):
                #print(epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                #print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                return
                #sys.exit()
        
        #parameters = {v.name: v for v in tf.trainable_variables() if v.name == 'beta/kernel:0' or v.name == 'beta/bias:0'}
        #emb = parameters['beta/kernel:0'].eval(vae.sess) + parameters['beta/bias:0'].eval(vae.sess)
        emb = [v for v in tf.trainable_variables() if v.name == 'beta/kernel:0'][0].eval(vae.sess)

        perplexity = evaluate(vae, emb, docs_va_h1, docs_va_h2, 'valid', config)
        #perplexity = evaluate(vae, emb, 'valid', summaries, writer, saver, vae.sess, epoch)
        # Display logs per epoch step
        #if epoch % display_step == 0:
        #    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        saver.save(vae.sess, os.path.join(save_path, 'model.ckpt'))

        tune.track.log(validation_perplexity=perplexity)

    return

def print_top_words(beta, feature_names, n_top_words=10):

    print('---------------Printing the Topics------------------')
    with open(save_path + '/topics.txt', 'w') as f:
        for i in range(len(beta)):
            print(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')

def evaluate(model, logit_beta, first_half, second_half, mode, config, docs_tr=None):
#def evaluate(model, logit_beta, step, summaries=None, writer=None, saver=None, session=None, epoch=None):
     
    beta = softmax(logit_beta, axis=1)
 
    #coherence = get_topic_coherence(beta, docs_tr, 'prodlda') if step == 'test' else np.nan
    #diversity = get_topic_diversity(logit_beta) if step == 'test' else np.nan
    
    theta = []
    #docs_h1 = docs_val_h1 if step == 'val' else docs_te_h1

    for base in range(0, first_half.shape[0], config['batch_size']):
        theta.append(softmax(model.topic_prop(first_half[base: min(base + config['batch_size'], first_half.shape[0])]), axis=1))
   
    theta = np.concatenate(theta, axis=0) 

    perplexity = get_perplexity(second_half, theta, beta)
    
    #if step == 'val':
 
        #weight_summaries = summaries.eval(session=session)
        #writer.add_summary(weight_summaries, epoch)
        #saver.save(session, ckpt + "/model.ckpt")

        #print("Model saved in path: %s" % save_path)
        #print('Epoch dev: {:d}'.format(epoch+1)) 
    
    if mode == 'test':
        
        #Storing the \beta during inference.
        with open(os.path.join(save_path, 'beta.pkl'), 'wb') as f:
            pickle.dump(logit_beta, f)

        ## get most used topics
        cnt = 0
        thetaWeightedAvg = np.zeros((1, int(t)))

        for base in range(0, docs_tr.shape[0], config['batch_size']):
            batch = docs_tr[base: min(base + config['batch_size'], docs_tr.shape[0])]
            sums = batch.sum(axis=1)
            cnt += sums.sum(axis=0)
            theta = softmax(model.topic_prop(batch), axis=1)
            weighed_theta = (theta.T * sums).T
            thetaWeightedAvg += weighed_theta.sum(axis=0)

        thetaWeightedAvg = thetaWeightedAvg.squeeze() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        print_top_words(beta, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])

        coherence = get_topic_coherence(beta, docs_tr, 'prodlda')
        diversity = get_topic_diversity(logit_beta)

        with open(save_path + '/' + mode + '_scores.csv', 'a') as handle:
            handle.write(str(perplexity) + ',' + str(coherence) + ',' + str(diversity) + '\n')

    return perplexity

#def calcPerp(model, step):
    
#    docs = docs_val if step == 'val' else docs_te
#    cost=[]

#    for doc in docs:
#        doc = doc.astype('float32')
#        n_d = np.sum(doc)
#        c=model.test(doc)
#        cost.append(c/n_d)

#    perplexity = np.exp(np.mean(np.array(cost)))
#    print('The approximated perplexity is: ', perplexity)
   
#    return perplexity

def get_config(topics):

    if topics == 50:

        current_best = {'layer1': 100,
                        'layer2': 100, 
                        'learning_rate': 0.004, 
                        'batch_size': 64,
                        'keep_prob': 0.4,
                        'epochs': 1176}

        space = {'layer1': 100,
                 'layer2': 100, 
                 'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)), 
                 'batch_size': 64,
                 'keep_prob': hp.loguniform('keep_prob', np.log(1e-12), np.log(1)),
                 'epochs': 10000}

    return [current_best], space

if __name__ == "__main__":

    #network_architecture,batch_size,learning_rate=make_network(f,s,t,b,r)

    #print(network_architecture)
    #print(opts)
    if mode == 'hyperparam':    

        ray.init()
        current_best, space = get_config(t) 

        algorithm = HyperOptSearch(space=space,
                                   metric="validation_perplexity", 
                                   mode="min",
                                   points_to_evaluate=current_best,
                                   random_state_seed=0)

        #algorithm = ConcurrencyLimiter(algorithm, max_concurrent=4)

        analysis = tune.run(train,
                            #metric="validation_perplexity",
                            #mode="min",
                            keep_checkpoints_num=1,
                            checkpoint_score_attr='min-validation_perplexity',
                            stop={"training_iteration": 1000},
                            resources_per_trial={"cpu": 2, "gpu": 0.33},
                            search_alg=algorithm,
                            num_samples=100)
                            #config=space)

        print("Best config is:", analysis.get_best_config(metric='validation_perplexity', mode='min'))
        with open(os.path.join(args.save_path, 'best_config.pkl'), 'wb') as f:
            pkl.dump(analysis.best_config, f)

    elif mode == 'train':
        
        with open(os.path.join(save_path, 'best_config.pkl'), 'rb') as f:
            config = pickle.load(f)

        scheduler = ASHAScheduler(time_attr='epochs',
                                  metric='validation_perplexity',
                                  mode='min',
                                  max_t=50,
                                  grace_period=50,
                                  reduction_factor=2,
                                  brackets=1)

        analysis = tune.run(train,
                            #metric="validation_perplexity",
                            #mode="min",
                            keep_checkpoints_num=1,
                            checkpoint_score_attr='min-validation_perplexity',
                            stop={"training_iteration": 1036},
                            resources_per_trial={"cpu": 2, "gpu": 0.33},
                            num_samples=1,
                            config=config,
                            scheduler=scheduler)

        #minibatches = create_minibatch(docs_tr.astype('float32'))
        #vae,emb = train(network_architecture, minibatches,m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate)

    else:

      with open(os.path.join(save_path, 'best_config.pkl'), 'rb') as f:
          config = pickle.load(f)

      vocab, docs_tr = load_data('train')
      vocab, docs_te_h1, docs_te_h2 = load_data('test')
      #config = tf.ConfigProto()
      #config.gpu_options.allow_growth=True
      #sess = tf.Session(config=config)

      network_architecture = {'n_hidden_recog_1': config['layer1'],
                            'n_hidden_recog_2': config['layer2'],
                            'n_hidden_gener_1': docs_tr.shape[1],
                            'n_input': docs_tr.shape[1],
                            'n_z': int(t)}
      vae = prodlda.VAE(network_architecture, learning_rate=config['learning_rate'], batch_size=config['batch_size'], keep_prob=config['keep_prob'])
      #if m=='prodlda':
      #    vae = prodlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
      #else:
      #    vae = nvlda.VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

      saver = tf.train.Saver()
      saver.restore(vae.sess, load_from)
      print("Model restored.")
     
      #emb = sess.run(vae.network_weights['weights_gener']['h2'])
      emb = [v for v in tf.trainable_variables() if v.name == 'beta/kernel:0'][0].eval(vae.sess)
      perplexity = evaluate(vae, emb, docs_te_h1, docs_te_h2, 'test', config, docs_tr)
      #calcPerp(vae, 'test')
