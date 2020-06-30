from typing import List, Optional


import torch
import transformers


model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model.to(torch.device('cuda:0'))

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")


def compare_losses(sentence1: str, sentence2: str):
    # crop to smaller sequence because longer sequence -> easier to predict
    tokenized_input1 = tokenizer.tokenize(sentence1)
    tokenized_input2 = tokenizer.tokenize(sentence2)
    length = min(len(tokenized_input1), len(tokenized_input2))
    tokenized_input1 = tokenized_input1[:length]
    tokenized_input2 = tokenized_input2[:length]
    tensor_input1 = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenized_input1)])[:model.config.n_positions]
    tensor_input2 = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenized_input2)])[:model.config.n_positions]
    if torch.cuda.is_available():
        tensor_input1 = tensor_input1.to(torch.device('cuda:0'))
        tensor_input2 = tensor_input2.to(torch.device('cuda:0'))
    loss1 = model(tensor_input1, labels=tensor_input1)[0]
    loss2 = model(tensor_input2, labels=tensor_input2)[0]
    return loss1.item(), loss2.item()


def _get_masked_loss(token_ids: List[int], next: Optional[List[int]]=None,
                     prev: Optional[List[int]]=None):
    if prev is None == next is None:
        raise ValueError('One and only one of prev and next should be specified.')
    if next is not None:
        labels = token_ids + [-100] * len(next)
        token_ids = token_ids + next
    else:
        labels = [-100] * len(prev) + token_ids
        token_ids = prev + token_ids
    token_ids = torch.tensor(token_ids)[:model.config.n_positions]
    labels = torch.tensor(labels)[:model.config.n_positions]
    if torch.cuda.is_available():
        token_ids = token_ids.to(torch.device('cuda:0'))
        labels = labels.to(torch.device('cuda:0'))
    return model(token_ids, labels=labels)[0].item()
        

def compare_losses(cand1, cand2, prev=None, next=None):
    if prev is None == next is None:
        raise ValueError('One and only one of prev and next should be specified.')

    ctx = prev if next is None else next
    ids_context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ctx))
    ids_cand1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand1))
    ids_cand2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand2))
    if prev is not None:
        loss1 = _get_masked_loss(ids_cand1, prev=ids_context)
        loss2 = _get_masked_loss(ids_cand2, prev=ids_context)
    else:
        loss1 = _get_masked_loss(ids_cand1, next=ids_context)
        loss2 = _get_masked_loss(ids_cand2, next=ids_context)
    return loss1 - loss2
