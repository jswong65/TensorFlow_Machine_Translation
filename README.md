# Neural Machine Translation with tensorflow

In this project I implemented a sequence-to-sequence neural machine translation model to convert sentences in English into sentences in French.

The details of the project can be found in the [project report](https://github.com/jswong65/TensorFlow_Machine_Translation/blob/master/project_report.pdf).

[baseline_model](https://github.com/jswong65/TensorFlow_Machine_Translation/blob/master/baseline_model.ipynb) is a vanilla seq2seq.

[attention_model](https://github.com/jswong65/TensorFlow_Machine_Translation/blob/master/attention_model.ipynb) leverages the bidirectional RNN for the encorder and the attention mechanism for the decoding process.

### Data Preprocessing
* The first step is to ensure all of the words is in lower case since a word in either upper case or lower case would be considered as the same word.
* In this dataset, the input sentences and target sentences have different length. To perform mini-batch training we need to pad <PAD> both the input sentences and target sentences to ensure the input sentences have the same length in a batch as the same for target sentences.
* <GO> and <EOS> are prepended and appended to each target sentence, respectively. Those tags help the model to understand the start and the end of a target sentence. Such tags are important for both training and inference purpose. Additionally, <UNK> tag will be used for the terms not in the training data.
* The text is then converted into the corresponding ids (one-hot-encoding) since most of the machine learning libraries cannot take text type of data as the input.
* To obtain the better representation of a word, word embedding is applied to transform a word into a vector. Word embedding has the capability to capture the syntactic and the semantic meaning of a word. A trainable matrix (with the shape [vocabulary size, embedding size]) was created to convert a word into the vector representation. vocabulary size indicates the number of unique terms in a corpus and embedding size implies the size of a word vector. Since the dataset contains sentences in two different languages (corpora), two embedding matrices are used to convert English words and French words into the corresponding word vectors.

### Sequence-to-Sequence model
A Sequence-to-Sequence (seq2seq) model is often leveraged to build a model for language translation and chat bot development. The model comprise two major components, the encoder and the decoders. Through the encoder-decoder mechanism, the model is able to learn the representation of input sequences with the encoder and generate the desired output sequences with the decoder. In contrast to the traditional statistical machine translation models which require the domain knowledge in the involved languages, seq2seq model is relatively intuitive to ed built and trained.

#### Encoder
I employed a multilayer bidirectional RNN for the implementation of the encoder since the bidirectionality on the encoder often produces the better performance. The main difference between a unidirectional RNN and a bidirectional RNN is that a bidirectional RNN maintains two hidden state, one for left-to-right propagation and the other for right-to-left propagation.

#### Decoder
Two types of decoders were built for diferent purpose, one for training and the other for inference. The training decoder is used for training purpose, and it takes the target sequence as the input to facilitate the training process (see Figure 5(a) - from Seq2Seq intro10). The inference decoder is exploited to generate the translated sentence, and the output of previous timestep is utilized as the input of current timestep Figure 5(b). A densely connected layer is added on the top of the decoders to generate the predicted word in each timestep. To improve the performance, the attention mechanism [6] is utilized that allows the decoders to obtain the most relevant memories from the encoder to enhance the decoding process.

### Sample Results
**English**: your least liked fruit is the pear , but their least liked is the banana .<br />
**French** (translated): votre moins aime fruit est la poire , mais leur moins aime est la mangue .<br />
**French** (target): votre moins aime fruit est la poire , mais leur moins aime est la banane .<br />

**English**: california is usually busy during spring , but it is never rainy in january .<br />
**French** (translated): californie est generalement occupe au printemps, mais jamais des pluies en janvier .<br />
**French** (target): californie est generalement occupe au printemps ,mais jamais des pluies en janvier .<br />

**English**: the apple is my most loved fruit , but the grape is his most loved .<br/>
**French** (translated): la pomme est le fruit le plus mon cher , mais le raisin est le plus aime<br/>
**French** (target): la pomme est le fruit le plus mon cher , mais le raisin est le plus aime .<br/>
