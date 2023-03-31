from libllama import LLaMAContext


def get_last_word_of_text(text, debug=False):
    text = text.replace('\n[EOS]', ' [EOS]')
    if text == '':
        return ''
    if debug:
        print(text.split(' '))
    return text.split(' ')[-1]


def make_chunks(data, chunk_size):
    while data:
        chunk, data = data[:chunk_size], data[chunk_size:]
        yield chunk


def print_batch(batch, ctx):
    for token in batch:
        print(ctx.llama_token_to_str(token).decode("utf-8"), end='')


def batch_to_str(batch, ctx):
    return ''.join([ctx.llama_token_to_str(token).decode("utf-8") for token in batch])


def create_generator(context: LLaMAContext, prompt: str, return_text=False):
    """
    Create a generator that yields the next token in the sequence.
    """
    prompt_to_tokens = context.llama_tokenize(prompt.encode("utf-8"), add_bos=True)

    for i, token in enumerate(prompt_to_tokens):
        context.llama_eval([token], i)

    while prompt_to_tokens[-1] != context.llama_token_eos():
        next_token = context.llama_sample_top_p_top_k(last_n_tokens=prompt_to_tokens)
        prompt_to_tokens += [next_token]

        yield context.llama_token_to_str(next_token).decode("utf-8") if return_text else next_token
        context.llama_eval([next_token], len(prompt_to_tokens) - 1)
        print(next_token)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
