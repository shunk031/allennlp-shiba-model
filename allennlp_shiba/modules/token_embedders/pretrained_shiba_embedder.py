from typing import Optional

import torch
import torch.nn as nn
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides
from shiba import Shiba
from shiba.model import get_pretrained_state_dict


@TokenEmbedder.register("shiba")
class PretrainedShibaEmbedder(TokenEmbedder):
    def __init__(
        self,
        downsampling_rate: int = 4,
        upsampling_kernel_size: int = 4,
        embedder_slice_count: int = 8,
        embedder_bucket_count: int = 16000,
        hidden_size: int = 768,
        local_attention_window: int = 128,
        deep_transformer_stack: Optional[nn.Module] = None,
        deep_transformer_requires_transpose: bool = True,
        attention_heads: int = 12,
        transformer_ff_size: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        padding_id: int = 0,
        max_length: int = 2048,
        shiba_specific_code: bool = False,
        deep_transformer_stack_layers: Optional[int] = None,
        train_parameters: bool = True,
        eval_mode: bool = False,
    ) -> None:
        super().__init__()

        shiba_model = Shiba(
            downsampling_rate=downsampling_rate,
            upsampling_kernel_size=upsampling_kernel_size,
            embedder_slice_count=embedder_slice_count,
            embedder_bucket_count=embedder_bucket_count,
            hidden_size=hidden_size,
            local_attention_window=local_attention_window,
            deep_transformer_stack=deep_transformer_stack,
            deep_transformer_requires_transpose=deep_transformer_requires_transpose,
            attention_heads=attention_heads,
            transformer_ff_size=transformer_ff_size,
            dropout=dropout,
            activation=activation,
            padding_id=padding_id,
            max_length=max_length,
            shiba_specific_code=shiba_specific_code,
            deep_transformer_stack_layers=deep_transformer_stack_layers,
        )
        shiba_model.load_state_dict(get_pretrained_state_dict())

        self.shiba_model = shiba_model
        self.config = self.shiba_model.config

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.shiba_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.shiba_model.eval()

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "deep_transformer":
                module.eval()
            else:
                module.train(mode)
        return self

    @overrides
    def forward(
        self, token_ids: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        output = self.shiba_model(input_ids=token_ids, attention_mask=~mask)
        return output["embeddings"]
