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
