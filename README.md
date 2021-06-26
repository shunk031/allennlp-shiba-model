# Allennlp Integration for [Shiba](https://github.com/octanove/shiba)

[![CI](https://github.com/shunk031/allennlp-shiba-model/actions/workflows/ci.yml/badge.svg)](https://github.com/shunk031/allennlp-shiba-model/actions/workflows/ci.yml)
[![Release](https://github.com/shunk031/allennlp-shiba-model/actions/workflows/release.yml/badge.svg)](https://github.com/shunk031/allennlp-shiba-model/actions/workflows/release.yml)
![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue?logo=python)
[![PyPI](https://img.shields.io/pypi/v/allennlp-shiba.svg)](https://pypi.org/project/allennlp-shiba/)

`allennlp-shiab-model` is a Python library that provides AllenNLP integration for [shiba-model](https://pypi.org/project/shiba-model/).

> SHIBA is an approximate reimplementation of CANINE [[1]](https://github.com/octanove/shiba#1) in raw Pytorch, pretrained on the Japanese wikipedia corpus using random span masking. If you are unfamiliar with CANINE, you can think of it as a very efficient (approximately 4x as efficient) character-level BERT model. Of course, the name SHIBA comes from the identically named Japanese canine.

## Installation

Installing the library and dependencies is simple using `pip`.

```shell
pip install allennlp-shiba
```

## Example

This library enables users to specify the in a jsonnet config file. Here is an example of the model in jsonnet config file:

```json
{
    "dataset_reader": {
        "tokenizer": {
            "type": "shiba",
        },
        "token_indexers": {
            "tokens": {
                "type": "shiba",
            }
        },
    },
    "model": {
        "shiba_embedder": {
            "type": "basic",
            "token_embedders": {
                "shiba": {
                    "type": "shiba",
                    "eval_model": true,
                }
            }

        }
    }
}
```


## Reference

- Joshua Tanner and Masato Hagiwara (2021). [SHIBA: Japanese CANINE model](https://github.com/octanove/shiba). GitHub repository, GitHub.

