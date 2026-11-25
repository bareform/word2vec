# Word-to-Vector

Word2Vec (Word-to-Vector) is method to learn dense, continuous vector representations from text.

## Installation

```console
git clone https://github.com/bareform/dependencies.git
conda update -n base -c defaults conda
conda env create -f dependencies/environment.yml
conda activate bareform

git clone https://github.com/bareform/word2vec.git
cd word2vec
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - Python 3.10
  - datasets
  - numpy
  - torch
    
</details>

## Quick Start

To train a `wikitext-2` CBOW Word2Vec:

```
python3 -m utils.trainer --config="./configs/wikitext2-cbow.toml"
```

To train a `wikitext-2` Skip-Gram Word2Vec:

```
python3 -m utils.trainer --config="./configs/wikitext2-skip-gram.toml"
```

You can download the pre-trained models [here](https://huggingface.co/luethan2025/word2vec) and use the provided Jupyter Notebook `inference.ipynb` to search semantically similar words. 

## Method

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean

Google

> We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.

## Citation

The original paper can be found at:
```
@misc{mikolov2013word2vec,
    title={Efficient Estimation of Word Representations in Vector Space},
    author={Tomas Mikolov and Kai Chen and Greg Corrado and Jeffrey Dean},
    year={2013},
    eprint={1301.3781},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
