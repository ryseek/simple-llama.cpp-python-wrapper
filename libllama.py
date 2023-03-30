import ctypes

DEFAULT_NUM_THREADS = 6

llama_token = ctypes.c_int
llama_token_p = ctypes.POINTER(llama_token)
llama_context_p = ctypes.c_void_p


class LLaMAContext:
    def __init__(self, ctx, llama):
        super().__init__()

        self._llama = llama
        self._ctx = ctx

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._ctx is not None:
            self._llama.llama_free(self._ctx)

        self._ctx = None
        self._llama = None

    def llama_token_bos(self):
        return int(self._llama.llama_token_bos())

    def llama_token_eos(self):
        return int(self._llama.llama_token_eos())

    def llama_n_vocab(self):
        return int(self._llama.llama_n_vocab(self._ctx))

    def llama_n_ctx(self):
        return int(self._llama.llama_n_ctx(self._ctx))

    def llama_token_to_str(self, token):
        return self._llama.llama_token_to_str(self._ctx, token)

    def llama_tokenize(self, byte_str, add_bos=False):
        tokens = (llama_token * 2048)()
        token_count = self._llama.llama_tokenize(self._ctx, byte_str, tokens, len(tokens), add_bos)
        return tokens[0:token_count]

    def llama_eval(self, tokens, n_past, n_threads=DEFAULT_NUM_THREADS):
        tokens = (llama_token * len(tokens))(*tokens)
        if self._llama.llama_eval(self._ctx, tokens, len(tokens), n_past, n_threads) != 0:
            raise Exception('eval failed')

    def llama_sample_top_p_top_k(self, last_n_tokens, top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.10):
        last_n_tokens_count = len(last_n_tokens)
        last_n_tokens = (llama_token * last_n_tokens_count)(*last_n_tokens)

        return self._llama.llama_sample_top_p_top_k(self._ctx, last_n_tokens, last_n_tokens_count, top_k, top_p, temp,
                                                    repeat_penalty)

    def llama_print_timings(self):
        self._llama.llama_print_timings(self._ctx)


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_int),
        ("n_parts", ctypes.c_int),
        ("seed", ctypes.c_int),
        ("f16_kv", ctypes.c_bool),
        ("logits_all", ctypes.c_bool),
        ("vocab_only", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("embedding", ctypes.c_bool),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
    ]


class LibLLaMA:
    def __init__(self, library):
        super().__init__()

        self._llama = ctypes.cdll.LoadLibrary(library)

        self._llama.llama_context_default_params.restype = llama_context_params

        self._llama.llama_init_from_file.argtypes = [ctypes.c_char_p, llama_context_params]
        self._llama.llama_init_from_file.restype = llama_context_p

        self._llama.llama_free.argtypes = [llama_context_p]
        self._llama.llama_free.restype = None

        self._llama.llama_eval.argtypes = [
            llama_context_p,  # llama_context     & lctx,
            llama_token_p,    # const llama_token * tokens,
            ctypes.c_int,     # const int           n_tokens
            ctypes.c_int,     # const int           n_past,
            ctypes.c_int      # const int           n_threads
        ]

        self._llama.llama_eval.restype = ctypes.c_int

        self._llama.llama_tokenize.argtypes = [
            llama_context_p,  # struct llama_context * ctx
            ctypes.c_char_p,  # const char           * text
            llama_token_p,    # llama_token          * tokens
            ctypes.c_int,     # int                    n_max_tokens
            ctypes.c_bool     # bool                   add_bos
        ]

        self._llama.llama_tokenize.restype = ctypes.c_int

        self._llama.llama_n_vocab.argtypes = [llama_context_p]
        self._llama.llama_n_vocab.restype = ctypes.c_int

        self._llama.llama_n_ctx.argtypes = [llama_context_p]
        self._llama.llama_n_ctx.restype = ctypes.c_int

        self._llama.llama_get_logits.argtypes = [llama_context_p]
        self._llama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self._llama.llama_token_to_str.argtypes = [llama_context_p, llama_token]
        self._llama.llama_token_to_str.restype = ctypes.c_char_p

        self._llama.llama_token_bos.restype = llama_token

        self._llama.llama_token_eos.restype = llama_token

        self._llama.llama_sample_top_p_top_k.argtypes = [
            llama_context_p,  # llama_context     * ctx,
            llama_token_p,    # const llama_token * last_n_tokens_data
            ctypes.c_int,     # int                 last_n_tokens_size
            ctypes.c_int,     # int                 top_k
            ctypes.c_float,   # float               top_p
            ctypes.c_float,   # float               temp,
            ctypes.c_float    # float               repeat_penalty
        ]
        self._llama.llama_sample_top_p_top_k.restype = llama_token

        self._llama.llama_print_timings.argtypes = [llama_context_p]

    def llama_context_default_params(self):
        return self._llama.llama_context_default_params()

    def llama_init_from_file(self, model_path, parameters=None):
        if parameters is None:
            parameters = self.llama_context_default_params()
        ctx = self._llama.llama_init_from_file(model_path.encode('utf-8'), parameters)
        if ctx == 0:
            raise Exception(f"Failed to load model from {model_path}")
        return LLaMAContext(ctx, self._llama)

