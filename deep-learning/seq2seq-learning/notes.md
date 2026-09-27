# Sequence to Sequence Learning for Machine Translation

In problem such as machine translation, where inputs and outputs each consist of variable length unaligned sequences, we generally rely on encoder-decoder architectures.

The special `<eos>` token marks the end of the sequence. Our model can stop making predictions once this token is generated. At the initial step of the RNN decoder there are two special design decisions to be aware of:

1. We begin every input with a special beginning-of-sequence `<bos>` token.
2. We may feed the final hidden state of the encoder into the decoder **at every single decoding time step**.

## 1. Teacher forcing

Running the input sequence on the encoder is easy, but handling the input output of the decoder requires more care. The most common approach is called _teacher forcing_.

### 1.1 How does it work?

Given an example where we have an exam and part **a** is needed for calculation in part **b**, and **b** is needed for part **c**, but if we get a wrong then the other parts are also wrong. What teacher forcing does is as follows: after we obtain an answer for part **a**, a teacher will compare our answer with the correct one, record the score for part **a**, and tell us the correct answer so that we can use it for part **b**.

Let's take a senario where we want to train an image captioning model, and the ground truth caption for the above image is "Two people reading a book".

- without _teacher forcing_, we would feed “birds” back to our RNN to predict the 3rd word.
- if we use Teacher Forcing, we would feed “people” to our RNN for the 3rd prediction, after computing and recording the loss for the 2nd prediction.

### What are the Pros and Cons of Teacher Forcing

**Pros**
Training with Teacher Forcing converges faster. At the early stages of training, the predictions of the model are very bad. If we do not use Teacher Forcing, the hidden states of the model will be updated by a sequence of wrong predictions, errors will accumulate, and it is difficult for the model to learn from that.

**Cons**
During inference, since there is usually no ground truth available, the RNN model will need to feed its own previous prediction back to itself for the next prediction. Therefore there is a discrepancy between training and inference, and this might lead to poor model performance and instability. `This is known as Exposure Bias in literature`.

## Further Reading

1. [Sequence to sequence learning with neural networks](https://arxiv.org/pdf/1409.3215).
2.

## Learning Resources

1. [Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html).
2. [What is Teacher Forcing?](https://medium.com/data-science/what-is-teacher-forcing-3da6217fed1c).
