import math
import torch
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import rotate_half
from functools import partial
from flash_attn import flash_attn_func


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


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


def generate_decoder_mask(num_querys, num_keys, dtype, device, debug=False):
    assert num_querys <= num_keys
    mask = torch.full((1,1,num_querys,num_querys), torch.finfo(dtype).min, device=device, dtype=torch.float32).triu(diagonal=1).type(dtype)
    prefix = torch.zeros((1,1,num_querys,num_keys-num_querys), device=device, dtype=dtype)
    mask = torch.cat([prefix, mask], dim=-1)

    assert mask.shape == (1, 1, num_querys, num_keys)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask[0,0].cpu())
        plt.savefig("mask.jpg", dpi=300)
        import IPython; IPython.embed(header='In generate_decoder_mask')

    assert (mask != 0).sum().item() == num_querys * (num_querys - 1) / 2
    assert (mask == 0).sum().item() == num_querys * num_keys - num_querys * (num_querys - 1) / 2

    return mask


def check_and_apply_beacon_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons):
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


# def generate_beacon_mask(num_ordinal, num_memory, num_beacons, dtype, device, layer_id, debug=False):
#     mask = torch.full(
#         (1, 1, num_ordinal + num_memory, num_beacons + num_ordinal + num_memory), 
#         torch.finfo(dtype).min, 
#         dtype=torch.float32, 
#         device=device
#     )

#     mask[0,0,:,:num_beacons].fill_(0)
#     mask[0,0,:num_ordinal,num_beacons:num_ordinal+num_beacons].triu_(diagonal=1)
#     mask[0,0,num_ordinal:,num_beacons+num_ordinal:].fill_diagonal_(0)
#     mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].fill_(0)

#     mask = mask.type(dtype)

#     for i in range(num_memory):
#         start = (i + 1) * (num_ordinal // num_memory) + num_beacons
#         end = num_ordinal + num_beacons
#         mask[0,0,num_ordinal+i, start: end] = torch.finfo(dtype).min

#     if debug and layer_id == 0:
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.imshow(mask[0,0].float().cpu().to(torch.float32))
#         plt.savefig("mask.jpg", dpi=300)
#         import IPython; IPython.embed(header='in generate_encoder_mask')

#     return mask


def generate_beacon_mask(num_ordinal, num_memory, num_beacons, dtype, device, layer_id, memory_mask, debug=False):
    mask = torch.full(
        (1, 1, num_ordinal + num_memory, num_beacons + num_ordinal + num_memory), 
        torch.finfo(dtype).min, 
        dtype=torch.float32, 
        device=device
    )

    mask[0,0,:num_ordinal,:num_beacons].fill_(0)

    if memory_mask == "triu":
        mask[0,0,num_ordinal:,:num_beacons].triu_(diagonal=1)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "diag":
        mask[0,0,num_ordinal:,:num_beacons].fill_diagonal_(0)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "full":
        mask[0,0,num_ordinal:,:num_beacons].fill_(0)
        mask[0,0,num_ordinal:,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    elif memory_mask == "mixed":
        for i in range(num_memory):
            start = 0
            end = i * 2 + 2
            mask[0,0,num_ordinal+i,start:end].fill_(0)
    else:
        raise NotImplementedError()
    
    mask[0,0,:num_ordinal,num_beacons:num_beacons+num_ordinal].triu_(diagonal=1)
    mask[0,0,num_ordinal:,num_beacons+num_ordinal:].fill_diagonal_(0)

    mask = mask.type(dtype)

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


def do_beacon_attn(query, key, value, cos, sin, o_proj, num_ordinal, num_memory, num_beacons, layer_id, memory_mask):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]
    
    Q, K, V = check_and_apply_beacon_rope(query, key, value, cos, sin, num_ordinal, num_memory, num_beacons)

    mask = generate_beacon_mask(num_ordinal, num_memory, num_beacons, dtype=query.dtype, device=query.device, layer_id=layer_id, memory_mask=memory_mask)

    score = Q @ K.transpose(-1,-2) / math.sqrt(128)
    score = score + mask
    attn = torch.softmax(score, dim=-1, dtype=torch.float32).type(score.dtype)

    output = attn @ V
    output = output.transpose(1,2).flatten(2)

    return o_proj(output)


def do_causal_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_full_flash_attn(query, key, value, cos, sin, out_proj: torch.nn.Linear = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = check_and_apply_rope(query, key, value, cos, sin)
    Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_causal_flash_attn_without_rope(query, key, value, out_proj: torch.nn.Linear = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=True
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_full_flash_attn_without_rope(query, key, value, out_proj: torch.nn.Linear = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


def do_adapter_attn(query, key, value, out_proj: torch.nn.Linear = None):
    batch_size, num_heads, num_query, head_dim = query.shape
    Q, K, V = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)

    attn_output = flash_attn_func(
        Q, K, V, causal=False
    )
    attn_output = attn_output.reshape(batch_size, num_query, num_heads * head_dim).contiguous()
    
    if out_proj is not None:
        attn_output = out_proj(attn_output)

    return attn_output


class Adapter(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.adapter = torch.nn.Parameter(torch.randn((1,4,4096), **kwargs) * 1e-3, requires_grad=True) 


class ProjectHead(torch.nn.Module):
    def __init__(self, layer, zero_init=False):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.key_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)

        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

        if zero_init:
            self.key_proj.weight.data.fill_(0)
            self.val_proj.weight.data.fill_(0)

    
    def get_lora_parameters(self):
        return [
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]


    def forward(self, activation: torch.Tensor):
        cache_k = self.key_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        cache_v = self.val_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        return cache_k, cache_v
    

class CrossAttnQKVProj(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.que_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.key_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)

        self.que_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

    def get_lora_parameters(self):
        return [
            self.que_proj.lora_A.default.weight,
            self.que_proj.lora_B.default.weight,
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]

    def forward(
            self, 
            hidden_states: torch.Tensor,
            memory_states: torch.Tensor
        ):
        query = self.que_proj(hidden_states).unflatten(-1, (32, 128)).transpose(1,2)
        key = self.key_proj(memory_states).unflatten(-1, (32, 128)).transpose(1,2)
        value = self.val_proj(memory_states).unflatten(-1, (32, 128)).transpose(1,2)
        return query, key, value
    

class QKVProj(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.que_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.key_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.val_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)

        self.que_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        self.key_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        self.val_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()

    def get_lora_parameters(self):
        return [
            self.que_proj.lora_A.default.weight,
            self.que_proj.lora_B.default.weight,
            self.key_proj.lora_A.default.weight,
            self.key_proj.lora_B.default.weight,
            self.val_proj.lora_A.default.weight,
            self.val_proj.lora_B.default.weight
        ]

    
    def forward(self, activation: torch.Tensor):
        query = self.que_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        key = self.key_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        value = self.val_proj(activation).unflatten(-1, (32, 128)).transpose(1,2)
        return query, key, value
    

class OProj(torch.nn.Module):
    def __init__(self, layer, zero_init=False):
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }
        self.out_proj = torch.nn.Linear(4096, 4096, bias=False, **kwargs)
        self.out_proj.weight.data = layer.self_attn.o_proj.weight.data.clone()
        
        if zero_init:
            self.out_proj.weight.data.fill_(0)

    def forward(self, activation: torch.Tensor):
        if activation.ndim == 4:
            activation = activation.transpose(1,2).flatten(2)
        output = self.out_proj(activation)
        return output


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, layer, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        kwargs = {
            "device": layer.self_attn.q_proj.weight.data.device,
            "dtype": layer.self_attn.q_proj.weight.data.dtype,
        }

        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)