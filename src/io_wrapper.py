import types
from abc import ABC
from dataclasses import dataclass
from typing import List


from src.criterion import get_criterion
from functools import partial
import random


import torch


def get_io_wrapper(wrapper_name, **kwargs):
    wrapper_name = wrapper_name.lower().replace(" ", "")

    if wrapper_name == "segmentrecurrentiowrapper":
        return SegmentRecurrentIOWrapper(**kwargs)
    else:
        raise NotImplementedError


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


def drop_tuple(x):
    return x[0] if isinstance(x, (tuple, list)) else x


@dataclass
class WrapperOutput:
    inputs: dict = None
    compute_loss: types.FunctionType = None

    def __getitem__(self, idx):
        return (
            self.inputs,
            self.compute_loss,
        )[idx]


class SegmentRecurrentIOWrapper(ABC):
    def __init__(self, tokenizer, chunk_size, truncation):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.task_type = None
        self.truncation = truncation


    def wrap(self, data: dict) -> List[WrapperOutput]:
        """

        auto encoding
        -------------
            * prompt : str
            * response : str
            * task_type : "auto encoding"

        language modeling
        -----------------
            * text : str
            * task_type : "language_modeling"

        question answering
        ------------------
            * prompt : str
            * response : str
            * train_prompt : bool
            * task_type : "question answering"

        perplexity
        ----------
            * text : str
            * task_type : "perplexity"

        news qa summation
        -----------------
            * disturb : str
            * context : str
            * questions : List[str]
            * answers : List[str]
            * summation : str
            * method : str
            * task_type : "news qa summation"

        memory utilization
        ------------------
            * text : str
            * task_type : "memory utilization"

        activation beacons
        ------------------
            * input_ids : List[int]
            * labels : List[int]
            * attention_mask : List[int]
            * distill : bool
            * length : int
            * task_type : "activation beacons"

        longdata-corpus copy
        --------------------
            * text : str
            * task_type : longdata copy
        """

        self.task_type = drop_tuple(data["task_type"])

        if self.task_type == "auto encoding":
            prompt = drop_tuple(data["prompt"])
            response = drop_tuple(data["response"])
            return self.wrap_ae_task(prompt, response)
        
        elif self.task_type == 'language modeling':
            text = drop_tuple(data["text"])
            return self.wrap_lm_task(text, self.truncation)
        
        elif self.task_type == 'question answering':
            prompt = drop_tuple(data["prompt"])
            response = drop_tuple(data["response"])
            train_prompt = drop_tuple(data["train_prompt"])
            return self.wrap_qa_task(prompt, response, train_prompt, self.truncation)
        
        elif self.task_type == 'perplexity':
            text = drop_tuple(data["text"])
            return self.wrap_ppl_task(text, self.truncation)
        
        elif self.task_type == 'news qa summation':
            disturb = drop_tuple(data['disturb'])
            context = drop_tuple(data['context'])
            questions = [drop_tuple(x) for x in data["questions"]]
            answers = [drop_tuple(drop_tuple(x)) for x in data["answers"]]
            summation = drop_tuple(data["summation"])
            method = drop_tuple(data['method'])
            return self.wrap_news_qa_sum(disturb, context, questions, answers, summation, method, self.truncation)

        elif self.task_type == 'news qa summation v2':
            disturb = drop_tuple(data['disturb'])
            context = drop_tuple(data['context'])
            questions = [drop_tuple(x) for x in data["questions"]]
            answers = [drop_tuple(drop_tuple(x)) for x in data["answers"]]
            summation = drop_tuple(data["summation"])
            return self.wrap_news_qa_sum_v2(disturb, context, questions, answers, summation, self.truncation)
        
        elif self.task_type == 'memory utilization':
            text = drop_tuple(data['text'])
            return self.wrap_mem_util_task(text, self.truncation)
        
        elif self.task_type == 'random memory utilization':
            text = drop_tuple(data['text'])
            return self.wrap_rnd_mem_util_task(text, self.truncation)
        
        elif self.task_type == 'activation beacons':
            input_ids = [x.item() for x in data['input_ids']]
            labels = [x.item() for x in data['labels']]
            attention_mask = [x.item() for x in data['attention_mask']]
            length = data['length'].item()
            distill = data['distill'].item()
            return self.wrap_activation_beacons(input_ids, labels, attention_mask, length, distill)
        elif self.task_type == 'longdata copy':
            text = drop_tuple(data['text'])
            return self.wrap_longdata_copy_task(text)
        else:
            raise NotImplementedError(self.task_type)

            
    def wrap_ae_task(self, prompt: str, response: str) -> List[WrapperOutput]:
        prompt = self.tokenizer(prompt, truncation=False, return_tensors='pt')
        response = self.tokenizer(response, truncation=False, return_tensors='pt')

        assert prompt.input_ids.shape[1] == self.chunk_size
        assert response.input_ids.shape[1] == self.chunk_size - 1

        ce_loss = get_criterion("ce")
        def compute_ce_loss(outputs, labels):
            logits = outputs.logits.cpu()
            logits = logits[:,:-1,:]
            return ce_loss(labels, logits)

        prompt = WrapperOutput(
            inputs={"input_ids": prompt.input_ids, "prefix_repeat_token": False},
            compute_loss=None
        )

        response = WrapperOutput(
            inputs={"input_ids": response.input_ids, "prefix_repeat_token": True},
            compute_loss=partial(compute_ce_loss, labels=response.input_ids)
        )

        return [prompt, response]


    def wrap_qa_task(self, prompt: str, response: str, train_prompt: bool, truncation: int) -> List[WrapperOutput]: 
        prompt = self.tokenizer(prompt, truncation=False, return_tensors='pt')
        response = self.tokenizer(response, truncation=False, return_tensors='pt')

        if truncation is None:
            truncation = 1e8

        if prompt.input_ids.shape[1] + response.input_ids.shape[1] > truncation:
            valid_length = truncation - response.input_ids.shape[1]
            prompt.input_ids = torch.cat([
                prompt.input_ids[:,:valid_length // 2],
                prompt.input_ids[:,-valid_length // 2:],
            ], dim=1)

        prompt.attention_mask = (
            torch.ones_like(prompt.input_ids) 
            if train_prompt
            else torch.zeros_like(prompt.input_ids)
        )

        input_ids = torch.cat([prompt.input_ids, response.input_ids], dim=-1)
        attention_mask = torch.cat([prompt.attention_mask, response.attention_mask], dim=-1)

        ce_loss = get_criterion("ce")
        def compute_ce_loss(outputs, labels, mask):
            logits = outputs.logits.cpu()
            logits = logits[:,:-1,:]
            return ce_loss(labels, logits, mask)

        result = []
        chunker = partial(segment, dim=-1, n=self.chunk_size)
        for chunk_id, (chunk, mask) in enumerate(zip(chunker(input_ids), chunker(attention_mask))):

            cond1 = chunk_id > 0
            cond2 = mask.sum().item() > 0

            compute_loss = (
                partial(compute_ce_loss, labels=chunk[:,1:], mask=mask[:,1:])
                if cond1 and cond2 else None
                )

            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk},
                compute_loss=compute_loss,
            )

            result.append(chunk_output)

        return result


    def wrap_lm_task(self, text: str, truncation: int) -> List[WrapperOutput]:
        text = self.tokenizer(text, truncation=False, return_tensors='pt')
        input_ids = text.input_ids[:,:truncation]

        ce_loss = get_criterion("ce")
        def compute_ce_loss(outputs, labels):
            logits = outputs.logits.cpu()
            logits = logits[:,:-1,:]
            return ce_loss(labels, logits)
        
        result = []
        for chunk in segment(input_ids, dim=-1, n=self.chunk_size):

            compute_loss = (partial(compute_ce_loss, labels=chunk[:,1:])
                            if chunk.shape[1] > 1 else None)

            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk},
                compute_loss=compute_loss,
            )

            result.append(chunk_output)

        return result
    

    def wrap_ppl_task(self, text: str, truncation: int):
        text = self.tokenizer(text, truncation=False, return_tensors='pt')
        input_ids = text.input_ids[:,:truncation]

        def compute_ppl(outputs, input_ids):
            logits = outputs.logits.log_softmax(dim=-1)
            gold_indices = input_ids[:,1:]
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            return logprobs[1:]

        result = []
        for chunk in segment(input_ids, dim=-1, n=self.chunk_size):
            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk},
                compute_loss=partial(compute_ppl, input_ids=chunk)
            )
            result.append(chunk_output)
        
        return result


    def wrap_news_qa_sum(self, disturb, context, questions, answers, summation, method, truncation: int):
        prefix = """
            Below is a paragraph which is inserted with bunch of irrelated context among it.
            You must remember the information that you think is neccessary.
            Then, answer questions at the end.

        """

        disturb = self.tokenizer(disturb, truncation=False, add_special_tokens=False, return_tensors='pt')
        disturb_ids = disturb.input_ids

        context = prefix + context
        context = self.tokenizer(context, truncation=False, add_special_tokens=False, return_tensors='pt')
        context_ids = context.input_ids

        prefix = self.tokenizer(prefix, truncation=False, return_tensors='pt')
        prefix_ids = prefix.input_ids

        gain = random.randint(1, truncation // self.chunk_size)
        truncation = self.chunk_size * gain

        if context_ids.shape[-1] + prefix_ids.shape[-1] < truncation:
            extra_ids = disturb_ids[:,:truncation - context_ids.shape[-1] - prefix_ids.shape[-1]]
            pos = random.randint(0, extra_ids.shape[-1] - 1)
            context_ids = torch.cat([prefix_ids, extra_ids[:,:pos], context_ids, extra_ids[:,pos:]], dim=-1)
        else:
            context_ids = torch.cat([prefix_ids, context_ids], dim=-1)[:,:truncation]

        result = []
        for chunk in segment(context_ids, dim=-1, n=self.chunk_size):
            inputs = {"input_ids": chunk}
            if method == 'distill':
                inputs.update({"forward_teacher": True})
            chunk_output = WrapperOutput(inputs=inputs)
            result.append(chunk_output)

        kl_div_loss = get_criterion("kldiv")
        ce_loss = get_criterion("ce")
        def compute_kl_loss(outputs, mask):
            teacher_logits = outputs['teacher_outputs'].logits.cpu()
            student_logits = outputs['student_outputs'].logits.cpu()
            return kl_div_loss(teacher_logits, student_logits, mask)
        
        def compute_ce_loss(outputs, labels, mask):
            logits = outputs.logits.cpu()
            logits = logits[:,:-1,:]
            return ce_loss(labels, logits, mask)
        
        num_kv_cache = None

        for question, answer in zip(questions, answers):
            question = f"Question: {question}"
            answer = f"Answer: {answer}"

            question = self.tokenizer(question, truncation=False, return_tensors='pt')
            answer = self.tokenizer(answer, truncation=False, return_tensors='pt')

            input_ids = torch.cat([question.input_ids, answer.input_ids], dim=-1)[:,:self.chunk_size]
            mask = torch.cat([torch.zeros_like(question.input_ids), torch.ones_like(answer.input_ids)], dim=-1)[:,:self.chunk_size]
            inputs = {
                "input_ids": input_ids,
                "clear_cache": num_kv_cache,
                "do_not_compress": True
            }

            if method == 'distill':
                compute_loss = partial(compute_kl_loss, mask=mask)
                inputs.update({"forward_teacher": True})
            elif method == 'lm':
                compute_loss = partial(compute_ce_loss, labels=input_ids[:,1:], mask=mask)

            chunk_output = WrapperOutput(
                inputs=inputs,
                compute_loss=compute_loss
            )
            result.append(chunk_output)
            num_kv_cache = min(input_ids.shape[-1], self.chunk_size)

        prompt = f"The paragraph is over, now summarize the above article in few sentences."
        prompt = self.tokenizer(prompt, truncation=False, return_tensors='pt')
        summation = self.tokenizer(summation, truncation=False, return_tensors='pt')

        input_ids = torch.cat([prompt.input_ids, summation.input_ids], dim=-1)[:,:self.chunk_size]
        mask = torch.cat([torch.zeros_like(prompt.input_ids), torch.ones_like(summation.input_ids)], dim=-1)[:,:self.chunk_size]
        inputs = {
            "input_ids": input_ids,
            "clear_cache": num_kv_cache,
            "do_not_compress": True
        }

        if method == 'distill':
            compute_loss = partial(compute_kl_loss, mask=mask)
            inputs.update({"forward_teacher": True})
        elif method == 'lm':
            compute_loss = partial(compute_ce_loss, labels=input_ids[:,1:], mask=mask)

        chunk_output = WrapperOutput(
            inputs=inputs,
            compute_loss=compute_loss
        )
        result.append(chunk_output)

        return result


    def wrap_news_qa_sum_v2(self, disturb, context, questions, answers, summation, truncation: int):
        prefix = """
            Below is a paragraph, and only a part of this paragraph is valid content, 
            the rest is irrelevant. Please identify the valid parts, 
            remember them, and answer the question after the end of the paragraph.\n\n
        """

        disturb = self.tokenizer(disturb, truncation=False, add_special_tokens=False, return_tensors='pt')
        disturb_ids = disturb.input_ids

        context = prefix + context
        context = self.tokenizer(context, truncation=False, add_special_tokens=False, return_tensors='pt')
        context_ids = context.input_ids

        prefix = self.tokenizer(prefix, truncation=False, return_tensors='pt')
        prefix_ids = prefix.input_ids

        truncation = random.randint(3072, 4096 - 512)
        remain = 4096 - truncation

        if context_ids.shape[-1] + prefix_ids.shape[-1] < truncation:
            extra_ids = disturb_ids[:,:truncation - context_ids.shape[-1] - prefix_ids.shape[-1]]
            pos = random.randint(1, extra_ids.shape[-1] - 2)
            context_ids = torch.cat([prefix_ids, extra_ids[:,:pos], context_ids, extra_ids[:,pos:]], dim=-1)
        else:
            context_ids = torch.cat([prefix_ids, context_ids], dim=-1)[:,:truncation]

        result = []
        for chunk in segment(context_ids, dim=-1, n=self.chunk_size):
            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk, "forward_teacher": True},
            )
            result.append(chunk_output)

        kl_div_loss = get_criterion("kldiv")
        ce_loss = get_criterion("ce")
        def compute_kl_loss(outputs, labels, mask):
            teacher_logits = outputs['teacher_outputs'].logits.cpu()
            student_logits = outputs['student_outputs'].logits.cpu()
            kl_part = kl_div_loss(teacher_logits, student_logits, mask)
            student_logits = student_logits[:,:-1,:]
            ce_part = ce_loss(labels, student_logits)
            return kl_part * 0.9 + ce_part * 0.1
        
        num_kv_cache = None

        for question, answer in zip(questions, answers):
            question = f"\n\nQuestion: {question}"
            answer = f"\n\nAnswer: {answer}"

            question = self.tokenizer(question, truncation=False, return_tensors='pt')
            answer = self.tokenizer(answer, truncation=False, return_tensors='pt')

            input_ids = torch.cat([question.input_ids, answer.input_ids], dim=-1)[:,:remain]

            mask = torch.cat([torch.zeros_like(question.input_ids), torch.ones_like(answer.input_ids)], dim=-1)[:,:remain]
            inputs = {
                "input_ids": input_ids,
                "clear_cache": num_kv_cache,
                "do_not_compress": True
            }

            compute_loss = partial(compute_kl_loss, mask=mask, labels=input_ids[:,1:])
            inputs.update({"forward_teacher": True})

            chunk_output = WrapperOutput(
                inputs=inputs,
                compute_loss=compute_loss
            )
            result.append(chunk_output)
            num_kv_cache = min(input_ids.shape[-1], self.chunk_size)

        prompt = f"\n\nThe paragraph is over, now summarize the above article in few sentences."
        prompt = self.tokenizer(prompt, truncation=False, return_tensors='pt')
        summation = self.tokenizer(summation, truncation=False, return_tensors='pt')

        input_ids = torch.cat([prompt.input_ids, summation.input_ids], dim=-1)[:,:remain]
        mask = torch.cat([torch.zeros_like(prompt.input_ids), torch.ones_like(summation.input_ids)], dim=-1)[:,:remain]
        inputs = {
            "input_ids": input_ids,
            "clear_cache": num_kv_cache,
            "do_not_compress": True
        }

        compute_loss = partial(compute_kl_loss, mask=mask)
        inputs.update({"forward_teacher": True})

        chunk_output = WrapperOutput(
            inputs=inputs,
            compute_loss=compute_loss
        )
        result.append(chunk_output)

        return result
    

    def wrap_mem_util_task(self, text, truncation):
        text = self.tokenizer(text, truncation=False, return_tensors='pt')

        if text.input_ids.shape[1] > truncation:
            text.input_ids = text.input_ids[:,:truncation]
            text.attention_mask = text

        assert truncation % 2 == 0
        chunk_size = truncation // 2
        assert self.chunk_size == chunk_size

        kl_div_loss = get_criterion("kldiv")
        def compute_kl_loss(outputs, mask):
            teacher_logits = outputs['teacher_outputs'].logits.cpu()
            student_logits = outputs['student_outputs'].logits.cpu()
            return kl_div_loss(teacher_logits, student_logits, mask)

        context_ids = text.input_ids
        random_ids = torch.randint(0, self.tokenizer.vocab_size - 1, size=(1,chunk_size))

        context_inputs = WrapperOutput(inputs={"input_ids": context_ids, "forward_teacher": True})
        random_inputs = WrapperOutput(inputs={"input_ids": random_ids, "do_not_compress": True, "forward_teacher": True},
                                      compute_loss=partial(compute_kl_loss, mask=torch.ones_like(random_ids)))
        
        return (context_inputs, random_inputs)
    

    def wrap_activation_beacons(self, input_ids, labels, attention_mask, length, distill):

        # shift labels
        labels = labels[1:] + [-100]

        input_ids = torch.tensor(input_ids, dtype=torch.long)[None,:]
        labels = torch.tensor(labels, dtype=torch.long)[None,:]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)[None,:]

        labels.masked_fill_(attention_mask == 0, -100)
        assert attention_mask.count_nonzero() == attention_mask.numel()

        valid_token_num = (labels != -100).count_nonzero()

        def get_loss(outputs, valid_token_num):
            return outputs.loss / valid_token_num
        
        kl_div = get_criterion("kldiv")
        def compute_distillation_loss(outputs, chunk_mask):
            stu_out = outputs["student_output"]
            tea_out = outputs["teacher_output"]
            return kl_div(tea_out.logits, stu_out.logits, chunk_mask)

        result = []
        chunker = partial(segment, dim=-1, n=self.chunk_size)

        for chunk_input_ids, chunk_labels, chunk_mask in zip(
            chunker(input_ids), chunker(labels), chunker(attention_mask)
        ):
            input_kwargs = {
                "input_ids": chunk_input_ids,
                "labels": chunk_labels
            }
            if distill is True:
                input_kwargs.update({"teacher_forward": True})

            chunk_output = WrapperOutput(
                inputs=input_kwargs,
                compute_loss=(
                    partial(get_loss, valid_token_num=valid_token_num) 
                    if distill is False 
                    else partial(compute_distillation_loss, chunk_mask=chunk_mask)) 
            )
            result.append(chunk_output)

        return result
    
    def wrap_longdata_copy_task(self, text: str):
        text = self.tokenizer(text, truncation=False, return_tensors='pt')
        
        maximum_length = min(
            self.truncation, 
            text.input_ids.shape[-1] // self.chunk_size * self.chunk_size)
        input_ids = text.input_ids[:,:maximum_length]
        
        def get_loss(outputs):
            return outputs.loss / maximum_length

        chunker = partial(segment, dim=-1, n=self.chunk_size)

        result = []
        for chunk in chunker(input_ids):
            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk},
                compute_loss=None
            )
            result.append(chunk_output)

        labels = input_ids.clone()
        input_ids = torch.cat([torch.tensor(self.tokenizer.eos_token_id)[None,None], input_ids[:,:-1]], dim=-1)

        for chunk, chunk_labels in zip(chunker(input_ids), chunker(labels)):
            chunk_output = WrapperOutput(
                inputs={"input_ids": chunk, "labels": chunk_labels},
                compute_loss=get_loss
            )
            result.append(chunk_output)

        return result
