import pytest
from allennlp.common import Params
from allennlp.data.tokenizers import Tokenizer
from allennlp_shiba.common.testing import AllennlpShibaTestCase
from allennlp_shiba.data.tokenizers import ShibaCodepointTokenizer
from shiba import CodepointTokenizer


class TestCodepointTokenizer(AllennlpShibaTestCase):
    @pytest.mark.parametrize(
        "input_str",
        (
            "自然言語処理",
            "柴ドリル",
        ),
    )
    def test_tokenize(self, input_str: str) -> None:
        allennlp_tokenizer = ShibaCodepointTokenizer()
        original_tokenizer = CodepointTokenizer()

        allennlp_output = allennlp_tokenizer.tokenize(input_str)
        original_output = original_tokenizer.encode(input_str)

        allennlp_token_ids = list(map(lambda x: x.text_id, allennlp_output))
        original_token_ids = original_output["input_ids"].cpu().numpy().tolist()

        assert allennlp_token_ids == original_token_ids

    def test_from_params(self) -> None:
        params = Params.from_file(
            self.FIXTURES_ROOT / "data" / "tokenizers" / "codepoint_tokenizer.jsonnet"
        )
        tokenizer = Tokenizer.from_params(params)

        assert isinstance(tokenizer, ShibaCodepointTokenizer)
