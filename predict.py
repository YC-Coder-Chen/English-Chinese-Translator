#!/usr/bin/env python3

import mxnet as mx
import pickle
from model import Encoder, Decoder, beam_search_translate

with open(f"./data/in_vocab.pkl", "rb") as fp:
    in_vocab = pickle.load(fp) 
with open(f"./data/out_vocab.pkl", "rb") as fp:
    out_vocab = pickle.load(fp) 

embed_size, num_hiddens, num_layers, ctx = 200, 200, 3, mx.cpu()
attention_size, drop_prob = 20, 0.1


encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)

encoder.load_parameters('./data/params_encoder_180')
decoder.load_parameters('./data/params_decoder_180')


# testing
"should return 我无法做到"
input_seq = "I can't do it ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 他很穷"
input_seq = "He is poor ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 她很穷"
input_seq = "She is poor ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 你还在办公室吗"
input_seq = "Are you still at the office ?"
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 汤姆不能去那里，因为他有很多作业"
input_seq = "Tom can't go out because he has a lot of homework ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 实话说，我不赞同你"
input_seq = "Frankly speaking, I don't agree with you ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)

"should return 我们明天讨论问题"
input_seq = "We're going to discuss the problem tomorrow ."
beam_search_translate(encoder, decoder, input_seq, 20, ctx, 3, in_vocab, out_vocab)