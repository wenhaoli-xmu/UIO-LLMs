from abc import ABC, abstractmethod
import torch
from torch.nn import functional as F


def get_criterion(criterion):
    if criterion.lower() == 'mse':
        return MSE()
    elif criterion.lower() == 'kldiv':
        return DistillKL()
    elif criterion.lower() == 'ce':
        return CELoss()
    else:
        raise NotImplementedError(f"Criterion '{criterion}' is not supported.")


class Criterion(ABC):
    pass


class MSE(Criterion):
    def __init__(self):
        super().__init__()

    def __call__(self, teacher_logits, student_logits, attention_mask):
        teacher_logits *= attention_mask.unsqueeze(-1).to(teacher_logits.device)
        student_logits *= attention_mask.unsqueeze(-1).to(student_logits.device)

        return torch.nn.functional.mse_loss(
            teacher_logits.view(-1, teacher_logits.shape[-1]), 
            student_logits.view(-1, student_logits.shape[-1]), 
            reduction='mean')
    

class DistillKL(Criterion):
    def __init__(self, temperature=1):
        super().__init__()
        self.temp = temperature

    def __call__(self, teacher_logits, student_logits, attention_mask):

        teacher_logits = teacher_logits.to(student_logits.device)
        attention_mask = attention_mask.to(student_logits.device)

        teacher_probs = F.softmax(teacher_logits / self.temp, dim=-1)
        student_probs = F.softmax(student_logits / self.temp, dim=-1)
        teacher_probs = teacher_probs.chunk(teacher_probs.shape[0])
        student_probs = student_probs.chunk(student_probs.shape[0])

        attention_mask = attention_mask.chunk(attention_mask.shape[0])

        teacher_probs = [teacher_prob[0][mask[0].to(torch.bool)] for teacher_prob, mask in zip(teacher_probs, attention_mask)]
        student_probs = [student_prob[0][mask[0].to(torch.bool)] for student_prob, mask in zip(student_probs, attention_mask)]

        teacher_probs = torch.cat(teacher_probs)
        student_probs = torch.cat(student_probs)

        loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

        return loss


class CELoss(Criterion):
    def __init__(self):
        super().__init__()

    def __call__(self, labels, logits, attention_mask=None):

        logits = logits.to(torch.float32)

        if attention_mask is None:
            attention_mask = torch.ones_like(labels)

        assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1
        labels = labels.squeeze(0)
        logits = logits.squeeze(0)

        if (attention_mask == 0).sum().item() > 0:
            split_point = torch.where(attention_mask == 1)[1].min().item()
            labels = labels[split_point:]
            logits = logits[split_point:,:]

        valid_num = (attention_mask == 1).sum()
        loss = torch.nn.functional.cross_entropy(logits, labels.to(torch.long), reduction='none')
        loss = loss.sum() / valid_num
        
        return loss