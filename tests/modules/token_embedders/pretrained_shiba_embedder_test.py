from typing import Dict

import pytest
import torch
from allennlp.common import Params
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp_shiba.common.testing import AllennlpShibaTestCase
from allennlp_shiba.data.token_indexers import PretrainedShibaIndexer
from allennlp_shiba.data.tokenizers import ShibaCodepointTokenizer
from allennlp_shiba.modules.token_embedders import PretrainedShibaEmbedder
from shiba import CodepointTokenizer, Shiba, get_pretrained_state_dict


class TestPretrainedShibaEmbedder(AllennlpShibaTestCase):
    def plane_shiba_output(
        self, sentence1: str, sentence2: str
    ) -> Dict[str, torch.Tensor]:

        shiba_model = Shiba()
        shiba_model.load_state_dict(get_pretrained_state_dict())
        shiba_model.eval()  # disable dropout
        tokenizer = CodepointTokenizer()

        inputs = tokenizer.encode_batch([sentence1, sentence2])
        outputs = shiba_model(**inputs)

        return outputs["embeddings"]

    def test_pretrained_shiba_embedder(self):

        tokenizer = ShibaCodepointTokenizer()
        token_indexer = PretrainedShibaIndexer()

        sentence1 = "自然言語処理"
        tokens1 = tokenizer.tokenize(sentence1)
        expected_tokens1 = ["[CLS]", "自", "然", "言", "語", "処", "理"]
        assert [t.text for t in tokens1] == expected_tokens1

        sentence2 = "柴ドリル"
        tokens2 = tokenizer.tokenize(sentence2)
        expected_tokens2 = ["[CLS]", "柴", "ド", "リ", "ル"]
        assert [t.text for t in tokens2] == expected_tokens2

        vocab = Vocabulary()

        params_dict = {
            "token_embedders": {
                "shiba": {
                    "type": "shiba",
                    "eval_mode": True,
                }
            }
        }
        params = Params(params_dict)
        token_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=params)

        instance1 = Instance({"tokens": TextField(tokens1, {"shiba": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"shiba": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        shiba_vectors = token_embedder(tokens)
        plane_shiba_vectors = self.plane_shiba_output(sentence1, sentence2)

        assert torch.all(torch.eq(shiba_vectors, plane_shiba_vectors))

    def test_from_params(self) -> None:
        params = Params.from_file(
            self.FIXTURES_ROOT
            / "modules"
            / "token_embedders"
            / "pretrained_shiba_embedder.jsonnet"
        )
        token_embedder = TokenEmbedder.from_params(params)
        assert isinstance(token_embedder, PretrainedShibaEmbedder)
