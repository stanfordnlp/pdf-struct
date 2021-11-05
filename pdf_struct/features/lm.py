# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Calculate LM scores based using huggingface transformers
# FIXME: This is a really inefficient implementation that copies the model
# for every joblib job. Try deploy the model on a server in the future.
from typing import List, Optional


import torch
import transformers


model = None
tokenizer = None
model_lang = None


def _init_lm(lang):
    # this function is idempotent
    global model, tokenizer, model_lang
    if model_lang is not None:
        if model_lang != lang:
            raise ValueError(
                f'init_lm called with different langs ("{lang}" and {model_lang})'
                ' on a single execution.')
        return
    if lang == 'en':
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    elif lang == 'ja':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "rinna/japanese-gpt2-medium")
        tokenizer = transformers.T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    else:
        raise ValueError(f'init_lm only supports ja or en but {lang} was given.')
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device('cuda:0'))
    model_lang = lang


def _get_masked_loss(token_ids: List[int], context: List[int]):
    labels = [-100] * len(context) + token_ids
    token_ids = context + token_ids
    token_ids = torch.tensor(token_ids)[:model.config.n_positions]
    labels = torch.tensor(labels)[:model.config.n_positions]
    if torch.cuda.is_available():
        token_ids = token_ids.to(torch.device('cuda:0'))
        labels = labels.to(torch.device('cuda:0'))
    return model(token_ids, labels=labels)[0].item()
        

def compare_losses(lang: str, cand1, cand2, prev=None, next=None):
    _init_lm(lang)
    if prev is None == next is None:
        raise ValueError('One and only one of prev and next should be specified.')

    ctx = prev if next is None else next
    ids_context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ctx))
    ids_cand1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand1))
    ids_cand2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand2))
    if prev is not None:
        loss1 = _get_masked_loss(ids_cand1, ids_context)
        loss2 = _get_masked_loss(ids_cand2, ids_context)
    else:
        loss1 = _get_masked_loss(ids_context, ids_cand1)
        loss2 = _get_masked_loss(ids_context, ids_cand2)
    return loss1 - loss2
