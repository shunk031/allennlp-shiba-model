from typing import Dict, List, Optional

from allennlp.data.token_indexers.token_indexer import IndexedTokenList, TokenIndexer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp_shiba.data.tokenizers import ShibaCodepointTokenizer
from overrides import overrides


@TokenIndexer.register("shiba")
class PretrainedShibaIndexer(TokenIndexer):
    def __init__(
        self,
        namespace: str = "tags",
        max_length: Optional[int] = None,
        token_min_padding_length: int = 0,
    ) -> None:

        super().__init__(token_min_padding_length=token_min_padding_length)

        self._namespace = namespace
        self._allennlp_tokenizer = ShibaCodepointTokenizer()
        self._tokenizer = self._allennlp_tokenizer._tokenizer
        self._added_to_vocabulary = False

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = self._max_length - num_added_tokens

            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        if self._added_to_vocabulary:
            return

        vocab.add_transformer_vocab(self._tokenizer, self._namespace)

        self._added_to_vocabulary = True

    def _extract_token(self, tokens: List[Token]) -> List[int]:
        indices: List[int] = []
        for token in tokens:
            indices.append(
                token.text_id
                if token.text_id is not None
                else self._tokenizer.encode(token.text)
            )
        return indices

    def _postprocess_output(self, output: IndexedTokenList) -> IndexedTokenList:
        if self._max_length is not None:
            # TODO (shunk031): need to support for max_length
            raise NotImplementedError

        return output

    @overrides
    def count_vocab_items(
        self, token: Token, counter: Dict[str, Dict[str, int]]
    ) -> None:
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> IndexedTokenList:

        indices = self._extract_token(tokens)

        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
        }

        return self._postprocess_output(output)

    @overrides
    def indices_to_tokens(
        self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary
    ) -> List[Token]:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        token_ids = indexed_tokens["token_ids"]
        type_ids = indexed_tokens.get("type_ids")

        return [
            Token(
                text=vocabulary.get_token_from_index(token_ids[i], self._namespace),
                text_id=token_ids[i],
                type_id=type_ids[i] if type_ids is not None else None,
            )
            for i in range(len(token_ids))
        ]

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {"token_ids": [], "mask": []}
        return output
