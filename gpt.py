from typing import List, Optional, Union
from litgpt.model import GPT, batched_index_select, do_softcapping
from litgpt.config import Config

import torch
import torch.nn as nn
from functools import partial

from utils import lightweight_entropy_estimator, parzen_entropy_estimator, pca_entropy_estimator

config = Config.from_file("out/finetune/full_baseline/final/model_config.yaml")

class new_GPT(GPT):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0,
        get_hidden_states: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        hidden_states = []
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            if input_pos.dim() > 2:
                # otherwise, things go wrong in `apply_rope`
                raise ValueError(f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}")
            if input_pos.shape[-1] != T:
                raise ValueError(f"input_pos.shape[-1] = {input_pos.shape[-1]} != {T} = idx.shape[1], must be the same")
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if input_pos.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
            if input_pos_maxp1 is not None:
                # Shorten final dimension so it just covers all `input_pos` entries
                if input_pos_maxp1 > self.max_seq_length:
                    raise ValueError(f"Positions in 'input_pos' must be in [0,{self.max_seq_length})")
                mask = mask[..., :input_pos_maxp1]
        else:
            # unsqueeze to have a batch dimension
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
            # `cos`, `sin` have shape (1, T, config.rope_n_elem)
            mask = None  # defaults to causal mask
            input_pos_maxp1 = None

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd ** 0.5, dtype=x.dtype)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos, input_pos_maxp1)
            hidden_states.append(x)
        hidden_states = torch.stack(hidden_states, dim=1)
        x = self.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            if get_hidden_states:
                return [
                    (clamp_head(self.lm_head(x_i)), hidden_states)
                    for x_i in x.split(lm_head_chunk_size, dim=1)
                ]
            return [
                clamp_head(self.lm_head(x_i))
                for x_i in x.split(lm_head_chunk_size, dim=1)
            ]
        else:
            if get_hidden_states:
                return clamp_head(self.lm_head(x)), hidden_states
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)



def main():

    model = new_GPT(config)
    print("Model loaded")
    x = torch.randint(0, 50256, (1, 128))
    print(x.shape)
    x, h = model(x, get_hidden_states=True)
    print(parzen_entropy_estimator(model.transformer.h[-1].mlp.fc.weight))
    print(x.shape)
    print(len(h))
    print(h.shape)
    print(h[0][-1].shape)
    l = 6
    entropy = sum(lightweight_entropy_estimator(h[0][i]) for i in range(h[0].size(0))) / h[0].size(0)
    print(entropy)

if __name__ == "__main__":
    main()

