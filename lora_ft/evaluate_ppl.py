from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn 
from peft import PeftModel, PeftConfig 
from tqdm import tqdm
import sys 
import json
import time  
import os 
from time import time
import fnmatch

"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""

def evaluate_ppl(dataset_name, model, tokenizer, ctx_length, ignore_last=False):
	# max_length = model.seqlen 
	model_seqlen = ctx_length
	max_length = ctx_length
	stride = ctx_length

	if dataset_name == "wikitext":
		test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
		encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
		seq_len = encodings.input_ids.size(1)
	elif dataset_name == "ptb":
		testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
		encodings = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
		seq_len = encodings.input_ids.size(1)
	elif dataset_name == "c4":
		try:
			valdata = load_dataset(
				'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
			)
		except:
			print("Trying again but with a different config")
			valdata = load_dataset(
				'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
			)
		encodings = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
		# encodings = encodings.input_ids[:, :(256 * model.seqlen)]
		seq_len = 256 * model_seqlen

	nlls = []
	prev_end_loc = 0
	total_time, total_iters = 0, 0
	for begin_loc in tqdm(range(0, seq_len, stride)):
		end_loc = min(begin_loc + max_length, seq_len)
		trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
		if (trg_len != stride) and ignore_last:
			break

		input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
		target_ids = input_ids.clone()
		target_ids[:, :-trg_len] = -100

		with torch.no_grad():
			start_ = time()
			outputs = model(input_ids, labels=target_ids)
			total_time += (time() - start_)
			total_iters += 1

			neg_log_likelihood = outputs.loss

		nlls.append(neg_log_likelihood)

		prev_end_loc = end_loc
		if end_loc == seq_len:
			break

	ppl = torch.exp(torch.stack(nlls).mean())
	return ppl.item(), total_time / total_iters