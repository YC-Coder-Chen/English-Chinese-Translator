#!/usr/bin/env python3

import pickle
import mxnet as mx
from mxnet.gluon import data as gdata, loss as gloss
from model import load_data, preprocess_data, build_vocab, Encoder, Decoder, train

# load the data
data = load_data()

# clean the data and preprocess the data
max_length = 25 # define the max_length of the sample
in_tokens, out_tokens, in_seqs, out_seqs = preprocess_data(data, max_length)
in_vocab, in_data = build_vocab(in_tokens, in_seqs) # build vocab
out_vocab, out_data = build_vocab(out_tokens, out_seqs)

# save the in_vocab and out_vocab for prediction
with open(f"./data/in_vocab.pkl", "wb") as fp:
    pickle.dump(in_vocab, fp) 
with open(f"./data/out_vocab.pkl", "wb") as fp:
    pickle.dump(out_vocab, fp) 
dataset =  gdata.ArrayDataset(in_data, out_data)

# train the model
loss = gloss.SoftmaxCrossEntropyLoss()
batch_size = 64
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
embed_size, num_hiddens, num_layers, ctx = 200, 200, 3, mx.gpu()
attention_size, drop_prob, lr, num_epochs = 20, 0.1, 0.005, 180
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)
train(encoder, decoder, data_iter, lr, batch_size, num_epochs, loss, out_vocab, ctx)
