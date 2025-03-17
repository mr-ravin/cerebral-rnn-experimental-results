# Cerebral LSTM implementation in Pytorch
This repository contains experimental results and the comparitive study and implementation of `Cerebral LSTM`, presented in the paper "Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs". Research paper is published in SN Computer Science Springer Nature Journal.

**Paper Title:** Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs

**Author:** [Ravin Kumar](https://mr-ravin.github.io)

**Publication:** 14th March 2020

**Published Paper:** [click here](https://link.springer.com/article/10.1007/s42979-020-0101-1)

**PDF available on Research Gate:** [click here](https://www.researchgate.net/publication/340013877_Cerebral_LSTM_A_Better_Alternative_for_Single-_and_Multi-Stacked_LSTM_Cell-Based_RNNs)

**DOI:** [https://doi.org/10.1007/s42979-020-0101-1](https://doi.org/10.1007/s42979-020-0101-1)

**Github:** [https://github.com/mr-ravin/cerebral-rnn-experimental-results](https://github.com/mr-ravin/cerebral-rnn-experimental-results) 

---
## Cerebral LSTM Architecture:

![image](https://github.com/mr-ravin/cerebral-rnn-experimental-results/blob/master/CerebralLSTM.png?raw=true)

```
Uf(t) = σ(Wuf ⋅ [h(t − 1), x(t)] + buf)
Ui(t) = σ(Wui ⋅ [h(t − 1), x(t)] + bui)
UCtmp(t) = tanh (Wuc ⋅ [h(t − 1), x(t)] + buc)
UC(t) = Uf(t) ∗ UC(t − 1) + Ui(t) ∗ UCtmp(t)
Uo(t) = σ(Wuo ⋅ [h(t − 1), x(t)] + buo)
Lf(t) = σ(Wlf ⋅ [h(t − 1), x(t)] + blf)
Li(t) = σ(Wli ⋅ [h(t − 1), x(t)] + bli)
LCtmp(t) = tanh (Wlc ⋅ [h(t − 1), x(t)] + blc)
LC(t) = Lf(t) ∗ LC(t − 1) + Li(t) ∗ LCtmp(t)
Lo(t) = σ(Wlo ⋅ [h(t − 1), x(t)] + blo)
h(t) = Uo(t) ∗ tanh(UC(t)) + Lo(t) ∗ tanh(LC(t))
```

### Impact of Initialisation of Trainable Parameters in Cerebral LSTM
The initial value of trainable parameters of upper and lower parts have impact onnumber of epochs required to train Cerebral LSTM cell. Ideally, upper and lowerparts should not have same initial values for their trainable parameters. 

#### Identical initial trainable parameter values for upper and lower parts

  `Initial Symmetry`: Upper and lower parts of the Cerebral LSTM process inputs identically, leading to similar cell states Uc(t) and Lc(t).

  `Redundancy`: Initial representations of upper and lower parts are redundant, poten-tially under-utilizing the model’s capacity.

  `Gradients`: Early training updates are similar, but divergence may occur over time,leading to different feature extraction.

#### Different initial trainable parameter values for upper and lower parts

  `Diverse Learning`: Upper and lower parts of Cerebral LSTM immediately capture different aspects of the data, enhancing representation diversity.

  `Specialization`: Faster convergence and better utilization of the dual-path architec-ture, as each path can specialize in different features.

  `Performance`: Improved performance due to richer, non-redundant representationsfrom the start. 

#### Conclusion
Our proposed recurrent cell ‘Cerebral LSTM’ showed the ability to better understanddata and has easily outperformed both single LSTM and two-stacked LSTM basedrecurrent neural networks. Many variants of Cerebral LSTM can be designed usingavailable varieties of LSTM cells such as peephole LSTM. Further research work canbe conducted on designing Cerebral LSTM based stacked recurrent neural networksfor designing deep learning architectures for understanding time-series data. Otherrecurrent cells including gated recurrent units can also be analyzed after modifying itsinternal connections similar to our cerebral structure.


#### Cite as:
```
Kumar, R. Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs. 
SN COMPUT. SCI. 1, 85 (2020). https://doi.org/10.1007/s42979-020-0101-1
```

### Architecture Description:
- `Pytorch Implementation of Cerebral LSTM` is available in `Cerebral_LSTM/Cerebral_LSTM_Implementation_in_Pytorch.ipynb` file.

- For the training loss graphs present in the research paper, see the below structure:
```
|
|-data
|
|-loss_values
      |
      |
      |- 2stack_lstm.txt 
      |
      |- proposed_model.txt
      |
      |- single_lstm.txt
      
```
- 'data' directory contains dataset used for comparison.
- 'loss_values' directory contains record of training loss for each model to perform comparative analysis.

```
Copyright (c) 2019-2023 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
