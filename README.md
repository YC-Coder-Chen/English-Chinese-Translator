# English-Chinese-Translator
============

Welcome, this project is an English to Chinese translator based on a seq2seq translation model with attention mechanism using MXNET and Python 3.6. We also provide a trained model based on 3-layer GRU trained on the [Chinese (Mandarin) - English dataset](http://www.manythings.org/anki/). Besides, a beam-search translation function is also provided, which performs better than the traditional greedy search.  
This project is inspired by the interactive deep learning course [Dive into Deep Learning](https://d2l.ai/) and some of the codes in the project are also taken from the above course. Because of the limited dataset and the limited computing power we had, this current version of the project is more like a demo on how to create a simple English to Chinese translator. We are working on a larger dataset.


Data
------------

The provided dataset came from [Chinese (Mandarin) - English dataset](http://www.manythings.org/anki/). You can download a different dataset to create your own translator.


Model training
------------

User can modify the [train_model.py](/train_model.py) and run the file to train your model. But before you train the model, you should download a tranditional Chinese to Simplified Chinese convert, [contains two .py files](https://github.com/skydark/nstools/tree/master/zhtools) (langconv.py, zh_wiki.py) into the directory where the [train_model.py](/train_model.py) is located.  

The default optimizer is "Adam", users can also change the optimizer to "SGD" or other optimizers supported by MXNET in the [model.py](/model.py). More specific parameters details are provided in the file. Below is the setting parameters for the trained classifier.

```
# for the dataset
max_length = 25 # define the max_length of the sample

# for the model
embed_size = 200
num_hiddens = 200
num_layers = 3
attention_size = 20
drop_prob = 0.1

# for the training
batch_size = 64
ctx = mx.gpu()
lr = 0.005
num_epochs = 180
```

Model Predict
------------

User can apply the trained model to translate a given English sentence into Simplified Chinese in the [predict.py](/predict.py) file. Currently, the model is not performing quite well because of the limited data and the limit computing power we had. 

```
Some sample test:
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
```
