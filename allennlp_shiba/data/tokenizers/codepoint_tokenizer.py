from typing import List

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides
from shiba import CodepointTokenizer


@Tokenizer.register("shiba")
class ShibaCodepointTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = CodepointTokenizer()

    @overrides
    def tokenize(self, text: str) -> List[Token]:

        encoded_tokens = self._tokenizer.encode(text)
        token_ids = encoded_tokens["input_ids"]

        tokens = []
        for tensor_token_id in token_ids:
            token_id = tensor_token_id.item()
            tokens.append(
                Token(text=self._tokenizer.decode([token_id]), text_id=token_id)
            )

        return tokens
