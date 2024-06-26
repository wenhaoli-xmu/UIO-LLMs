from abc import ABC, abstractmethod
from typing import Any
import torch
import json
from typing import List
from functools import partial


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


class Modifier(ABC):

    def __init__(self, model, save_ckp, load_ckp):
        self.model = model
        self.load_ckp = load_ckp
        self.save_ckp = save_ckp

        if self.load_ckp is not None:
            self.load_checkpoint()

    def get_conf(self, config):
        if config is not None:
            with open(config, 'r') as f:
                conf = json.load(f)
            print("=" * 40 + " Config " + "=" * 40)
            print(json.dumps(conf, indent=4
                ).replace("\n    ", "\n"
                ).replace("{", ""
                ).replace("}", ""
                ).strip().replace('"', ''))
            print("=" * 88)
            self.conf = conf
        else:
            self.conf = None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
    
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @property
    def device(self):
        return self.model.device
    
    @property
    def dtype(self):
        return self.model.dtype
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_model(self):
        for param in self.ft_params():
            param.requires_grad_(True)
    
    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)

    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save(self.ft_params(), ckp)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @abstractmethod
    def ft_params(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class LayerWiseModifier(Modifier):
    def num_layers(self):
        return len(self.get_layers())

    def freeze_layer(self, layer_idx):
        layer = self.get_layers()[layer_idx]
        for param in layer.parameters():
            param.requires_grad_(False)

    def unfreeze_layer(self, layer_idx):
        for param in self.layerwise_ft_params(layer_idx):
            param.requires_grad_(True)
            
    @abstractmethod
    def get_layers(self):
        raise NotImplementedError

    @abstractmethod
    def layerwise_ft_params(self, layer_idx):
        raise NotImplementedError

    @abstractmethod
    def layerwise_input(self, layer_idx):
        raise NotImplementedError

    @abstractmethod
    def layerwise_output(self, layer_idx):
        raise NotImplementedError


class SegmentRecurrentModifier(Modifier):
    def __init__(self, model, save_ckp, load_ckp, chunk_size):
        super().__init__(model, save_ckp, load_ckp)
        self.chunk_size = chunk_size


    @abstractmethod
    def get_memories(self, segment_id):
        raise NotImplementedError
    
    
    @torch.no_grad()
    def generate_for_ae_task(
            self, 
            input_ids: torch.Tensor, 
            max_new_tokens: int = 256,
    ) -> List:
        assert input_ids.shape[1] == self.chunk_size

        self.model(input_ids=input_ids)
        logits = self.model(input_ids=torch.zeros((1,0), dtype=torch.long), prefix_repeat_token=True).logits
        inputs = torch.argmax(logits, dim=-1)
        while input_ids.shape[1] <= self.chunk_size + max_new_tokens:
            logits = self.model(input_ids=inputs).logits
            inputs = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, inputs.to(input_ids.device)], dim=-1)
        self.reset()

        return input_ids
    

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: List = [2],
    ) -> List:

        prompt_length = input_ids.shape[1]
        context = input_ids[:,:-1]
        chunker = partial(segment, dim=-1, n=self.chunk_size)

        for chunk_context_inputs in chunker(context):
            self.model(input_ids=chunk_context_inputs)
        
        new_token = input_ids[:,-1:]
        while input_ids.shape[1] < prompt_length + max_new_tokens:
            logits = self.model(input_ids=new_token).logits.cpu()
            new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, new_token.to(input_ids.device)], dim=1)

            # 在合适的时候进行打断
            if new_token.item() in eos_token_id:
                break

        self.reset()
        
        return input_ids
