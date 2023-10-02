# Cerebral LSTM implementation in Pytorch
This repository contains experimental results and the comparitive study and implementation of `Cerebral LSTM`, presented in the paper "Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs". Research paper is published in SN Computer Science Springer Nature Journal.

#### Paper Title: Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs

#### Author: [Ravin Kumar](https://mr-ravin.github.io)

#### Publication: 14th March 2020

#### View Published Paper: [click here](https://link.springer.com/article/10.1007/s42979-020-0101-1)

#### PDF available on Research Gate: [click here](https://www.researchgate.net/publication/340013877_Cerebral_LSTM_A_Better_Alternative_for_Single-_and_Multi-Stacked_LSTM_Cell-Based_RNNs)

#### Doi: https://doi.org/10.1007/s42979-020-0101-1

#### Cite as:
```
Kumar, R. Cerebral LSTM: A Better Alternative for Single- and Multi-Stacked LSTM Cell-Based RNNs. 
SN COMPUT. SCI. 1, 85 (2020). https://doi.org/10.1007/s42979-020-0101-1
```

### Architecture Description:
- `Pytorch Implementation of Cerebral LSTM` is available in `Cerebral_LSTM_Implementation_in_Pytorch.ipynb` file.

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
