import sys

from libllama import LibLLaMA, LLaMAContext
from helpers import print_batch, make_chunks, get_last_word_of_text, bcolors, batch_to_str, create_generator

model = ''
weights = ''

if __name__ == '__main__':
    lib = LibLLaMA(model)

    prompt = """ How many states in the USA? \n"""

    parameters = lib.llama_context_default_params()
    parameters.seed = -1
    parameters.n_ctx = 256
    parameters.n_parts = 1

    with lib.llama_init_from_file(weights, parameters) as ctx:
        for token in create_generator(ctx, prompt, return_text=True):
            print(token, end="")
            sys.stdout.flush()
