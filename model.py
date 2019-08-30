#!/usr/bin/env python3

"""
langconv is taken from the following github to convert traditional Chinese to simplified Chinese
https://github.com/skydark/nstools/blob/master/zhtools/langconv.py

"""

from langconv import * 
import collections
import io
import math
import jieba
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import nn, rnn

# define the padding tag, begin tag and end tag
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

"Load the Chinese-English data"
def load_data():
    with io.open('./data/cmn.txt') as f:
        lines = f.readlines()
    return lines

"Preprocess data into tokens"
def preprocess_data(data, max_length):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    for st in data:
        in_seq, out_seq = st.rstrip().split('\t')
        # apply langconv to convert traditional Chinese to simplified Chinese
        # apply jieba to cut Chinese text into words
        in_seq_tokens, out_seq_tokens = in_seq[:-1].lower().split(' '), [tk for tk in jieba.cut(Converter('zh-hans').convert(out_seq.lower().rstrip()))]
        marker = in_seq[-1]
        in_seq_tokens.append(marker)
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_length - 1:
            continue # skip the sentence if too long
        in_tokens.extend(in_seq_tokens)
        out_tokens.extend(out_seq_tokens)
        in_seqs.append(in_seq_tokens+ [EOS] + [PAD] * (max_length - len(in_seq_tokens) - 1))
        out_seqs.append(out_seq_tokens + [EOS] + [PAD] * (max_length - len(out_seq_tokens) - 1))
    return in_tokens, out_tokens, in_seqs, out_seqs

"Use the tokens bag to create vocab object and indices"
def build_vocab(token, seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(token),reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in seqs]
    return vocab, indices

"Define the Encoder model"
class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        """
        inputs: (batch_size, max_length)
        embedding: (max_length, batch_size, embed_size)
        state: [(num_layer, batch_size, num_hiddens)]
        output: (max_length, batch_size, num_hiddens)
        
        """
        embedding = self.embedding(inputs.T)
        return self.gru(embedding, state)

    def begin_state(self, *args, **kwargs):
        return self.gru.begin_state(*args, **kwargs)

"Define the attention model"
def attention_function(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False, flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model

"Define the function to perfrom attention mechanism"
def attention_forward(attention, enc_states, dec_state):
    """
    enc_states: (max_length, batch_size, num_hiddens)
    dec_state: (batch_size, num_hidden)
    
    """
    dec_state = dec_state.expand_dims(0)
    dec_states = nd.broadcast_axis(dec_state, axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    """
    enc_and_dec_states: (max_length, batch_size, 2*num_hiddens)
    attention(enc_and_dec_states): (max_length, batch_size, 1)
    alpha_prob: (max_length, batch_size, 1)
    
    """
    
    alpha_prob = nd.softmax(attention(enc_and_dec_states), axis=0)
    return (alpha_prob * enc_states).sum(axis=0)

"Define the Decoder model"
class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_function(attention_size)
        self.gru = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def forward(self, cur_input, state, enc_states):
        # Apply Attention Mechanism, use the last hidden state of encoder as enc_state
        """
        enc_states: (max_length, batch_size, num_hiddens)
        state: (max_length, batch_size, num_hiddens) # the dec_states
        context: (batch_size, num_hidden) # the weighted avg of enc_states and dec_state[-1]
        cur_input: (batch_size,)
        self.embedding(cur_input): (batch_size, embed_size)
        input_context: (batch_size, embed_size + num_hidden)
        
        """
        context = attention_forward(self.attention, enc_states, state[0][-1])
        input_context = nd.concat(self.embedding(cur_input), context, dim=1)
        output, state = self.gru(input_context.reshape(1,cur_input.shape[0], -1), state)
        output = self.dense(output).reshape(-1, self.vocab_size)
        return output, state

    def begin_state(self, enc_state):
        return enc_state # use the last state of enc_state as initial state

"Define the loss function, ignore all the padding mask"
def loss_func(encoder, decoder, inputs, outputs, loss, batch_size, out_vocab, ctx):
    batch_size = inputs.shape[0] #in case batch_size is changed
    # in each batch, we have to reinitialize the state
    enc_state = encoder.begin_state(batch_size=batch_size, ctx = ctx)
    enc_outputs, enc_state = encoder(inputs, enc_state) # encode finished
    dec_state = decoder.begin_state(enc_state= enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size, ctx=ctx) # inital input
    mask, num_valid_tokens = nd.ones(shape=(batch_size,), ctx = ctx), 0
    l = nd.array([0], ctx = ctx)
    for y_state in outputs.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y_state)).sum()
        dec_input = y_state # teacher forcing
        num_valid_tokens += mask.sum().asscalar()
        mask = mask * (y_state != out_vocab.token_to_idx[EOS])
    # decoding finish
    return l / num_valid_tokens

"Define the train function"
def train(encoder, decoder, data_iter, lr, batch_size, num_epochs, loss, out_vocab, ctx):
    encoder.initialize(ctx = ctx, init = init.Xavier(), force_reinit=True)
    decoder.initialize(ctx = ctx, init = init.Xavier(), force_reinit=True)    
    
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, y in data_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            y = y.astype('float32')
            with autograd.record():
                l = loss_func(encoder, decoder, X, y, loss, batch_size, out_vocab, ctx)
            l.backward()
            enc_trainer.step(1) # we already calculate the avg
            dec_trainer.step(1) # we already calculate the avg
            l_sum += l.asscalar()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}, loss {l_sum / len(data_iter)}")
            encoder.save_parameters(f'./data/params_encoder_{epoch + 1}')
            decoder.save_parameters(f'./data/params_decoder_{epoch + 1}')

def predict_rest(encoder, decoder, input_sq, max_length, idx, dec_state, enc_output, score, in_vocab, out_vocab, ctx):
    output_tokens = []
    pred_token = out_vocab.idx_to_token[idx]
    if pred_token == EOS:  # when find EOS, finished
        return [output_tokens, score]
    else:
        output_tokens.append(pred_token)
        dec_input = nd.array([1], ctx = ctx) * idx
        for _ in range(max_length - 1):
            dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
            pred = dec_output.argmax(axis=1)
            pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
            if pred_token == EOS:  # when find EOS, finished
                break
            else:
                output_tokens.append(pred_token)
                score = score * nd.softmax(dec_output[0])[idx].asscalar()
                dec_input = pred
        return [output_tokens, score]

def beam_search_translate(encoder, decoder, input_seq, max_length, ctx, beam_size, in_vocab, out_vocab):
    in_tokens = input_seq.lower().split(' ')
    in_tokens += [EOS] + [PAD] * (max_length - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)], ctx = ctx)
    enc_state = encoder.begin_state(batch_size=1, ctx = ctx)
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]], ctx = ctx)
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    # the first character prediction
    dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
    topk = nd.topk(dec_output, k=beam_size, ret_typ='indices').asnumpy().astype('int32')
    for idx in topk[0]:
        score = nd.softmax(dec_output[0])[idx].asscalar()
        sample_output = predict_rest(encoder, decoder, input_seq, max_length, idx, dec_state, enc_output, score, in_vocab, out_vocab, ctx)
        output_tokens.append(sample_output)
    
    for idx in range(len(output_tokens)):
        output_tokens[idx][1] = math.log(output_tokens[idx][1])/(len(output_tokens[idx][0]) ** 0.75)
    return output_tokens          
