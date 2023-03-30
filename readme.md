# Simple [llama.cpp](https://github.com/ggerganov/llama.cpp) Python Wrapper
This is a python wrapper around the LLama library, providing an easier-to-use API for accessing LLama's functionality.

## Installation
To use this wrapper, you'll need to build the LLama shared library first.
The `libllama.so` shared library can be built by adding this makefile targert to the LLama makefile:
```makefile
libllama.so: llama.o ggml.o
    $(CXX) $(CXXFLAGS) -g3 -shared -fPIC -o libllama.so llama.o ggml.o $(LDFLAGS)
```


## How to use the wrapper 
```python
import sys
from libllama import LibLLaMA

lib_path = 'path/to/libllama.so'
weights_path = 'path/to/weights'

prompt = "Here is how to build a python web server in 3 steps:\n"

with LibLLaMA(lib_path).llama_init_from_file(weights_path) as ctx:
    initial_prompt = ctx.llama_tokenize(prompt.encode("utf-8"), add_bos=True)

    for i, token in enumerate(initial_prompt):
        ctx.llama_eval([token], i)

    last_n_tokens = initial_prompt
    text = prompt
    print(text)
    while True:
        prediction = ctx.llama_sample_top_p_top_k(last_n_tokens=last_n_tokens)
        last_n_tokens += [prediction]

        str_token = ctx.llama_token_to_str(prediction).decode("utf-8")
        text += str_token

        print(str_token, end="")
        sys.stdout.flush()

        if prediction == ctx.llama_token_eos():
            break

        ctx.llama_eval([prediction], len(last_n_tokens) - 1)
```

The output of [gpt4all](https://github.com/nomic-ai/gpt4all) model:
```
Here is how to build a python web server in 3 steps:

1. Install the Flask framework on your computer with pip or conda.
2. Create a new flask application by running "flask serve" command, and open it in your browser.
3. Modify your HTML files or templates as needed for your specific requirements, such as adding Flask-Login for authentication
```