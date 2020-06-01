import torch
from transformers import *


# ================ EXAMPLE 1 ================

# Load model and tokenizer for LEGAL-BERT-FP on EU legislation
tokenizer = AutoTokenizer.from_pretrained('../models/bert-base-eu-100k')
lm_eurlex_bert = AutoModelWithLMHead.from_pretrained('../models/bert-base-eu-100k')

text_1 = 'Establish criteria to be met by farmers in order to fulfil the obligation to maintain an [MASK] area in a state suitable for grazing or cultivation'
input_ids = tokenizer.encode(text_1)
print(tokenizer.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'establish', 'criteria', 'to', 'be', 'met', 'by', 'farmers', 'in', 'order', 'to', 'fu', '##lf', '##il',
# 'the', 'obligation', 'to', 'maintain', 'an', '[MASK]', 'area', 'in', 'a', 'state', 'suitable', 'for', 'grazing',
# 'or', 'cultivation', '[SEP]']
outputs = lm_eurlex_bert(torch.tensor([input_ids]))[0]
print(tokenizer.convert_ids_to_tokens(outputs[0, 19].max(0)[1].item()))
# The top prediction for [MASK] is "agricultural"

# ================ EXAMPLE 2 ================
# Load model and tokenizer for LEGAL-BERT-FP on US contracts
tokenizer = AutoTokenizer.from_pretrained('../models/bert-base-contracts-500k')
lm_contracts_bert = AutoModelWithLMHead.from_pretrained('../models/bert-base-contracts-500k')

text_1 = 'The Participant may [MASK] this Agreement by giving the Service Provider at least one month’s30 days’ notice in writing'
input_ids = tokenizer.encode(text_1)
print(tokenizer.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'the', 'participant', 'may', '[MASK]', 'this', 'agreement', 'by', 'giving', 'the', 'service', 'provider',
# 'at', 'least', 'one', 'month', '’', 's', '##30', 'days', '’', 'notice', 'in', 'writing', '[SEP]']
outputs = lm_contracts_bert(torch.tensor([input_ids]))[0]
print(tokenizer.convert_ids_to_tokens(outputs[0, 4].max(0)[1].item()))
# The top prediction for [MASK] is "terminate"


# ================ EXAMPLE 3 ================
# Load model and tokenizer for LEGAL-BERT-FP on ECHR cases
tokenizer = AutoTokenizer.from_pretrained('../models/bert-base-echr-500k')
lm_contracts_bert = AutoModelWithLMHead.from_pretrained('../models/bert-base-echr-500k')

text_1 = 'The Zagreb County Court found the first applicant guilty as charged and sentenced the first applicant to three years’ [MASK].'
input_ids = tokenizer.encode(text_1)
print(tokenizer.convert_ids_to_tokens(input_ids))
# ['[CLS]', 'the', 'zagreb', 'county', 'court', 'found', 'the', 'first', 'applicant', 'guilty', 'as', 'charged',
# 'and', 'sentenced', 'the', 'first', 'applicant', 'to', 'three', 'years', '’', '[MASK]', '.', '[SEP]']
outputs = lm_contracts_bert(torch.tensor([input_ids]))[0]
print(tokenizer.convert_ids_to_tokens(outputs[0, 21].max(0)[1].item()))
# The top prediction for [MASK] is "imprisonment"
