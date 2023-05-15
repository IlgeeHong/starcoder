from transformers import AutoModelForCausalLM, AutoTokenizer
# added
import os
import torch
from pathlib import Path
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

with smp.tensor_parallelism():
    model = AutoModelForCausalLM.from_config("bigcode/starcoder_config")

# checkpoint = "bigcode/starcoder"
# device = "cuda" # for GPU usage or "cpu" for CPU usage

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # to save memory consider using fp16 or bf16 by specifying torch.dtype=torch.float16 for example
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))