# from src.misc import get_model_and_tokenizer
from src.misc import get_env_conf
from src.misc import Evaluator
import types
import torch
from typing import Optional, List


DEVICE_MAP = {
    "model.beacon_embed_tokens": 0,
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 1,
    "model.layers.11": 1,
    "model.layers.12": 1,
    "model.layers.13": 1,
    "model.layers.14": 1,
    "model.layers.15": 1,
    "model.layers.16": 2,
    "model.layers.17": 2,
    "model.layers.18": 2,
    "model.layers.19": 2,
    "model.layers.20": 2,
    "model.layers.21": 2,
    "model.layers.22": 2,
    "model.layers.23": 2,
    "model.layers.24": 3,
    "model.layers.25": 3,
    "model.layers.26": 3,
    "model.layers.27": 3,
    "model.layers.28": 3,
    "model.layers.29": 3,
    "model.layers.30": 3,
    "model.layers.31": 3,
    "model.norm": 3,
    "lm_head": 3
}


def _beacon_forward(self, 
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
        self.memory.reset()
        self.my_logits = []

        self.memory.prepare(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        while not self.memory.finish:
            input_ids, attention_mask, past_key_values, labels = self.memory.step()

            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                shift_labels=False,
            )

            self.memory.update_memory(outputs.past_key_values)

            if labels is not None:
                self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)

            logits = self.memory.output(outputs).logits
            self.my_logits.append(logits.cpu())

        # outputs = self.memory.output(outputs)
        outputs.logits = torch.cat(self.my_logits, dim=-2)

        return outputs


def reset(self):
    ...


def get_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from accelerate import load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download

    model_id = "namespace-Pt/activation-beacon-llama2-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_id, beacon_window=2048, beacon_stride=[2048], beacon_ratio=[32], trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = load_checkpoint_and_dispatch(model, snapshot_download(model_id), device_map=DEVICE_MAP)
    model.eval()

    return tokenizer, model


if __name__ == '__main__':
    test_conf = get_env_conf("test.json")

    tokenizer, model = get_model_and_tokenizer()
    model.chunk_size = 1000000
    model._beacon_forward = types.MethodType(_beacon_forward, model)
    model.reset = types.MethodType(reset, model)

    evaluator = Evaluator(model, tokenizer, eval=None, tasks=test_conf)
    evaluator.evaluate()
