from transformers import AutoModelForCausalLM, AutoTokenizer
# added
import os
import torch
from pathlib import Path
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

checkpoint = "bigcode/starcoder"
checkpoints = sorted(Path(checkpoint).glob("*.pth"))

print(checkpoints)

# def load(checkpoint: str, local_rank: int, world_size: int,):


# checkpoint = "bigcode/starcoder"
# device = "cuda" # for GPU usage or "cpu" for CPU usage

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # to save memory consider using fp16 or bf16 by specifying torch.dtype=torch.float16 for example
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))