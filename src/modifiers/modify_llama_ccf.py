import math, types, warnings
from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint

from transformers.models.llama.modeling_llama import (
    rotate_half,
    CausalLMOutputWithPast,
    CrossEntropyLoss,
    )
from transformers.cache_utils import Cache

from transformers.cache_utils import Cache
import torch.nn.functional as F

from src.modifier import Modifier, SegmentRecurrentModifier
from functools import partial

from flash_attn import flash_attn_func
from peft import get_peft_model, LoraConfig, TaskType
from copy import deepcopy
import builtins


def fake_print(*args, **kwargs):
    pass


def apply_rotary_pos_emb(mat, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    mat_embed = (mat * cos) + (rotate_half(mat) * sin)

    return mat_embed


def new_posid(num_token: int, device, dtype, bsz):
    appendix = torch.arange(num_token, device=device)
    appendix = appendix[None,:].expand(bsz, -1)
    return appendix


def check_and_apply_rope(query, key, value, cos, sin):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_kv)[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_kv))
    V = value

    return Q, K, V


def check_and_apply_encoder_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]


    assert key.shape == (batch_size, num_heads, num_kv, head_dim)
    assert value.shape == (batch_size, num_heads, num_kv, head_dim)

    assert num_query == num_ordinal + num_memory
    assert num_kv == num_ordinal + num_memory + num_beacons

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)

    if num_memory > 0:
        ordinal_query = apply_rotary_pos_emb(query[:,:,:-num_memory,:], cos, sin, new_posid_spec(num_ordinal) + num_beacons)
        ordinal_key = apply_rotary_pos_emb(key[:,:,:-num_memory,:], cos, sin, new_posid_spec(num_beacons + num_ordinal))
        cover_tokens = num_ordinal // num_memory
        memory_query = apply_rotary_pos_emb(query[:,:,-num_memory:,:], cos, sin, (new_posid_spec(num_memory) + 1) * cover_tokens + num_beacons)
        memory_key = apply_rotary_pos_emb(key[:,:,-num_memory:,:], cos, sin, (new_posid_spec(num_memory) + 1) * cover_tokens + num_beacons)
        Q = torch.cat([ordinal_query, memory_query], dim=-2)
        K = torch.cat([ordinal_key, memory_key], dim=-2)
    else:
        Q = apply_rotary_pos_emb(query, cos, sin, new_posid_spec(num_ordinal) + num_beacons)
        K = apply_rotary_pos_emb(key, cos, sin, new_posid_spec(num_beacons + num_ordinal))

    V = value

    return Q, K, V


def generate_encoder_mask(num_ordinal, num_memory, num_beacons, dtype, device, layer_id, debug=False):
    mask = torch.full(
        (1, 1, num_ordinal + num_memory, num_beacons + num_ordinal + num_memory), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:,:num_beacons].fill_(0)
    mask[0,0,:num_ordinal,num_beacons:num_ordinal+num_beacons].triu_(diagonal=1)
    mask[0,0,num_ordinal:,num_beacons+num_ordinal:].fill_diagonal_(0)
    mask = mask.type(dtype)

    mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].fill_(0)
    for i in range(num_memory):
        start = (i + 1) * (num_ordinal // num_memory) + num_beacons
        end = num_ordinal + num_beacons
        mask[0,0,num_ordinal+i, start: end] = torch.finfo(dtype).min

    if debug and layer_id == 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].float().cpu().to(torch.float32))
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='in generate_encoder_mask')

    return mask


def do_encoder_attn(query, key, value, cos, sin, o_proj, num_ordinal, num_memory, layer_id):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]
    
    Q, K, V = check_and_apply_encoder_rope(query, key, value, cos, sin, num_ordinal, num_memory, 0)

    mask = generate_encoder_mask(num_ordinal, num_memory, 0, dtype=query.dtype, device=query.device, layer_id=layer_id)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)


def do_causal_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear):
    dtype = query.dtype
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True 
    )

    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    attn_output = out_proj(attn_output)
    return attn_output



def teacher_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )     

        bsz, q_len = hidden_states.shape[:2]

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # NOTE: the kv caches are used for inference
            if hasattr(self, "k_cache"):
                key_states = torch.cat([self.k_cache, key_states], dim=-2)
                value_states = torch.cat([self.v_cache, value_states], dim=-2)
            self.k_cache = key_states.detach()
            self.v_cache = value_states.detach()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=4096)

        attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def encoder_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len = hidden_states.shape[:2]
        num_ordinal = q_len - self.num_memory

        self.memory.append(hidden_states[:,-self.num_memory:,:])

        # q_len = oridnal + memory
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=4096)

        attn_output = do_encoder_attn(
            query_states, 
            key_states, 
            value_states, 
            cos, sin,
            self.o_proj, 
            num_ordinal, self.num_memory,
            self.layer_idx)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def decoder_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = hidden_states @ self.k_proj.base_layer.weight.T
        value_states = hidden_states @ self.v_proj.base_layer.weight.T

        if hasattr(self, "k_cache"):
            key_states = torch.cat([self.k_cache, key_states], dim=-2)
            value_states = torch.cat([self.v_cache, value_states], dim=-2)
        self.k_cache = key_states.detach()
        self.v_cache = value_states.detach()

        if len(self.memory_detach) > 0:
            memory_states = torch.cat(self.memory_detach, dim=-2)
            memory_k = self.k_proj(memory_states)
            memory_v = self.v_proj(memory_states)

            key_states = torch.cat([memory_k, key_states], dim=-2)
            value_states = torch.cat([memory_v, value_states], dim=-2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=4096)
        attn_output = do_causal_flash_attn(query_states, key_states, value_states, cos, sin, self.o_proj)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


def encoder_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return None


def decoder_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        loss, _, valid_token_num = compute_loss(logits, labels, shift=False)
        print(f"my loss: {loss.item()}", flush=True)
        loss = loss * valid_token_num

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits)


class Teacher(Modifier):
    def __init__(self, model):
        super().__init__(model, None, None)
    
    def reset(self):
        raise NotImplementedError
    
    def ft_params(self):
        raise NotImplementedError
    

class Encoder(torch.nn.Module):
    def __init__(self, encoder, num_memory, eos_token_embed):
        super().__init__()
        self.encoder = encoder
        self.num_memory = num_memory
        self.retrieval_token = torch.nn.Parameter(eos_token_embed[None,None,:], requires_grad=True)

        for layer in encoder.base_model.model.model.layers:
            layer.self_attn.forward = types.MethodType(encoder_attn_forward, layer.self_attn)
            layer.self_attn.memory = []
            layer.self_attn.num_memory = num_memory

    def ft_params(self):
        params = [self.retrieval_token]
        for layer in self.encoder.base_model.model.model.layers:
            params += [
                layer.self_attn.q_proj.lora_A.default.weight,
                layer.self_attn.q_proj.lora_B.default.weight,
                layer.self_attn.v_proj.lora_A.default.weight,
                layer.self_attn.v_proj.lora_B.default.weight,
                ]
        return params
    
    def forward(self, input_ids):
        inputs_embeds = self.encoder.base_model.model.model.embed_tokens(input_ids).cpu()
        inputs_embeds = torch.cat([inputs_embeds, self.retrieval_token.expand(-1,self.num_memory,-1)], dim=1)
        self.encoder(inputs_embeds=inputs_embeds)


class Decoder(torch.nn.Module):
    def __init__(self, decoder, eos_token_embed):
        super().__init__()
        self.decoder = decoder
        self.repeat_token = torch.nn.Parameter(eos_token_embed[None,None,:], requires_grad=True)
        for layer in decoder.base_model.model.model.layers:
            layer.self_attn.forward = types.MethodType(decoder_attn_forward, layer.self_attn)
            layer.self_attn.memory_detach = []


    def ft_params(self):
        params = [self.repeat_token]
        for layer in self.decoder.base_model.model.model.layers:
            params += [
                layer.self_attn.k_proj.lora_A.default.weight,
                layer.self_attn.k_proj.lora_B.default.weight,
                layer.self_attn.v_proj.lora_A.default.weight,
                layer.self_attn.v_proj.lora_B.default.weight,
                ]
        return params


    def forward(self, input_ids, labels=None, cat_repeat_token=False):
        inputs_embeds = self.decoder.base_model.model.model.embed_tokens(input_ids).cpu()
        if cat_repeat_token:
            inputs_embeds = torch.cat([self.repeat_token.to(inputs_embeds.device), inputs_embeds], dim=1)
        outputs = self.decoder(inputs_embeds=inputs_embeds, labels=labels)
        return outputs
    

class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, teacher, chunk_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher = teacher
        self.chunk_size = chunk_size
        self.accum_input_ids = None


    def ft_params(self):
        return self.encoder.ft_params() + self.decoder.ft_params()
    

    def reset(self):
        self.accum_input_ids = None
        for layer in self.decoder.decoder.base_model.model.model.layers:
            layer.self_attn.memory_detach = []
            if hasattr(layer.self_attn, "k_cache"):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache
        for layer in self.encoder.encoder.base_model.model.model.layers:
            layer.self_attn.memory = []
        for layer in self.teacher.model.layers:
            if hasattr(layer.self_attn, "k_cache"):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache

    
    def reset_except_memory(self):
        self.accum_input_ids = None
        for layer in self.decoder.decoder.base_model.model.model.layers:
            if hasattr(layer.self_attn, "k_cache"):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache


    def clear_last_cache(self, num_kv_cache):
        for layer in self.decoder.decoder.base_model.model.model.layers:
            if hasattr(layer.self_attn, "k_cache"):
                layer.self_attn.k_cache = layer.self_attn.k_cache[:,:-num_kv_cache,:]
                layer.self_attn.v_cache = layer.self_attn.v_cache[:,:-num_kv_cache,:]
        for layer in self.teacher.model.layers:
            if hasattr(layer.self_attn, "k_cache"):
                layer.self_attn.k_cache = layer.self_attn.k_cache[:,:-num_kv_cache,:]
                layer.self_attn.v_cache = layer.self_attn.v_cache[:,:-num_kv_cache,:]



    def transfer_kv_cache(self):

        for encoder_layer, decoder_layer in zip(
            self.encoder.encoder.base_model.model.model.layers,
            self.decoder.decoder.base_model.model.model.layers
        ):
            
            if decoder_layer.self_attn.k_cache.shape[-2] == self.chunk_size:
                del decoder_layer.self_attn.k_cache
                del decoder_layer.self_attn.v_cache
            else:
                decoder_layer.self_attn.k_cache = decoder_layer.self_attn.k_cache[:,:-self.chunk_size,:]
                decoder_layer.self_attn.v_cache = decoder_layer.self_attn.v_cache[:,:-self.chunk_size,:]

            memory = encoder_layer.self_attn.memory[-1]
            memory_detach = memory.detach()
            memory_detach.requires_grad_(True)
            decoder_layer.self_attn.memory_detach.append(memory_detach)


    def forward(
            self, 
            input_ids,
            labels=None, 
            show_debug_message=False, 
            prefix_repeat_token=False, # used in copy task
            forward_teacher=False, # used in qa summation task
            clear_cache=None,  # used in qa summation task
            do_not_compress=False, # used in memory utilization task
        ):
        if clear_cache is not None and isinstance(clear_cache, int):
            self.clear_last_cache(clear_cache)

        assert input_ids.shape[1] <= self.chunk_size
        print = builtins.print if show_debug_message else fake_print

        print("=" * 80)
        print(f"In EncDec forward function")
        print(f"\t* input_ids: {input_ids.shape}")
        print(f"\t* prefix_repeat_token: {prefix_repeat_token}")
        print(f"\t* clear_cache: {clear_cache}")
        print(f"\t* do_not_compress: {do_not_compress}")
        print(f"\t* forward_teacher: {forward_teacher}\n")

        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.memory_detach.__len__()}")
        print(f"\t\tlen(kv_cache): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.base_model.model.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(teacher kv cache): {self.teacher.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.teacher.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(accum_input_ids): {self.accum_input_ids.shape[1] if self.accum_input_ids is not None else 0}")

        print(f"\tActions:")

        outputs = self.decoder(
            input_ids=input_ids,
            labels=labels,
            cat_repeat_token=prefix_repeat_token
        )

        if forward_teacher is True:
            with torch.no_grad():
                teacher_logits = self.teacher(
                    input_ids=input_ids
                ).logits

        # NOTE: this part is used in memory utilization task
        if do_not_compress is True:
            if forward_teacher is True:
                return {
                    "teacher_outputs": CausalLMOutputWithPast(logits=teacher_logits),
                    "student_outputs": outputs
                }
            else:
                return outputs

        self.accum_input_ids = (
            torch.cat([self.accum_input_ids, input_ids], dim=-1) 
            if self.accum_input_ids is not None 
            else input_ids)

        print(f"\t\t{input_ids.shape[1]} tokens newly come in\n")
        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.memory_detach.__len__()}")
        print(f"\t\tlen(kv_cache): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.base_model.model.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(teacher kv cache): {self.teacher.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.teacher.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(accum_input_ids): {self.accum_input_ids.shape[1] if self.accum_input_ids is not None else 0}")
        print(f"\tActions:")
          
        while self.accum_input_ids.shape[1] >= self.chunk_size:
            input_ids = self.accum_input_ids[:,:self.chunk_size]
            self.accum_input_ids = self.accum_input_ids[:,self.chunk_size:]

            self.encoder(input_ids)
            self.transfer_kv_cache()
            
            print(f"\t\tCompression occured!")

        print()
        print(f"\tCurrent State:")
        print(f"\t\tlen(memory): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.memory_detach.__len__()}")
        print(f"\t\tlen(kv_cache): {self.decoder.decoder.base_model.model.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.decoder.decoder.base_model.model.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(teacher kv cache): {self.teacher.model.layers[0].self_attn.k_cache.shape[-2] if hasattr(self.teacher.model.layers[0].self_attn, 'k_cache') else 0}")
        print(f"\t\tlen(accum_input_ids): {self.accum_input_ids.shape[1] if self.accum_input_ids is not None else 0}")
        print("", flush=True)

        if forward_teacher is True:
            return {
                "teacher_outputs": CausalLMOutputWithPast(logits=teacher_logits),
                "student_outputs": outputs
            }
        else:
            return outputs


class CCF(SegmentRecurrentModifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        self.get_conf(config)

        self.chunk_size = self.conf['chunk_size']
        self.lora_rank = self.conf['lora_rank']
        self.lora_alpha = self.conf['lora_alpha']
        self.lora_dropout = self.conf['lora_dropout']
        self.num_memory = self.conf['num_memory']

        encoder = deepcopy(model)
        decoder = deepcopy(model)
        teacher = model

        for layer in teacher.model.layers:
            layer.self_attn.forward = types.MethodType(teacher_attn_forward, layer.self_attn)

        encoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=['q_proj', 'v_proj']
        )
        decoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=['k_proj', 'v_proj']
        )
        encoder = get_peft_model(encoder, encoder_peft_config)
        decoder = get_peft_model(decoder, decoder_peft_config)

        encoder.base_model.model.forward = types.MethodType(encoder_model_forward, encoder.base_model.model)
        decoder.base_model.model.forward = types.MethodType(decoder_model_forward, decoder.base_model.model)

        eos_token_embed = encoder.base_model.model.model.embed_tokens.weight[2,:]
        encoder = Encoder(encoder, num_memory=self.num_memory, eos_token_embed=eos_token_embed)
        decoder = Decoder(decoder, eos_token_embed=eos_token_embed)
        encoder_decoder = EncoderDecoder(encoder, decoder, teacher, self.chunk_size)

        super().__init__(encoder_decoder, save_ckp, load_ckp, chunk_size=self.chunk_size)

    def ft_params(self):
        return self.model.ft_params()

    def reset(self):
        self.model.reset()

    def get_memories(self, segment_id):
        states = []
        for layer in self.model.encoder.encoder.base_model.model.model.layers:
            states += [
                layer.self_attn.memory[segment_id].cpu()
            ]
        states = torch.cat(states, dim=0)

        if self.model.decoder.decoder.base_model.model.model.layers[0].self_attn.memory_detach[segment_id].grad is not None:
            grads = []
            for layer in self.model.decoder.decoder.base_model.model.model.layers:
                grads += [
                    layer.self_attn.memory_detach[segment_id].grad.data.cpu()
                ]
            grads = torch.cat(grads, dim=0)
        else:
            grads = torch.zeros_like(states)
        
        return grads, states

