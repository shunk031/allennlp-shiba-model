import pytest
from allennlp.data.vocabulary import Vocabulary
from allennlp_shiba.common.testing import AllennlpShibaTestCase
from allennlp_shiba.data.token_indexers import PretrainedShibaIndexer
from allennlp_shiba.data.tokenizers import ShibaCodepointTokenizer
from shiba import CodepointTokenizer


class TestPretrainedShibaIndexer(AllennlpShibaTestCase):
    @pytest.mark.parametrize(
        "input_str",
        ("自然言語処理", "柴ドリル"),
    )
    def test_pretrained_shiba_indexer(self, input_str: str):
        tokenizer = CodepointTokenizer()
        allennlp_tokenizer = ShibaCodepointTokenizer()

        tokens = tokenizer.encode(input_str)
        allennlp_tokens = allennlp_tokenizer.tokenize(input_str)

        vocab = Vocabulary()
        indexer = PretrainedShibaIndexer()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab)

        output_token_ids = indexed["token_ids"]
        expect_token_ids = tokens["input_ids"].cpu().numpy().tolist()

        output_mask = indexed["mask"]
        expect_mask = (~tokens["attention_mask"].cpu().numpy()).tolist()

        assert output_token_ids == expect_token_ids
        assert output_mask == expect_mask
