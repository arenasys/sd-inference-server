import lark
import torch

class WeightedTree(lark.Tree):
    pass

prompt_grammar = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasis | deemphasis | numeric | scheduled | alternate | plain | WHITESPACE)*
emphasis: "(" prompt ")"
deemphasis: "[" prompt "]"
numeric: "(" prompt ":" [_WHITESPACE] NUMBER [_WHITESPACE]")"
scheduled: "[" [prompt ":"] prompt ":" [_WHITESPACE] NUMBER [_WHITESPACE]"]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
_WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""", tree_class=WeightedTree)

def parse_prompt(prompt, steps):
    if not prompt:
        return [(steps, [["", 1.0]])]

    def extract(tree, step):
        def propagate(node, output, step, weight):
            if type(node) == WeightedTree:
                node.weight = weight
                children = node.children
                if node.data == "emphasis": node.weight *= 1.1
                if node.data == "deemphasis": node.weight /= 1.1
                if node.data == "numeric":
                    node.weight *= float(node.children[1])
                    children = [node.children[0]]
                if node.data == "scheduled":
                    if step <= float(node.children[2]):
                        children = [node.children[0]]
                    else:
                        children = [node.children[1]]
                if node.data == "alternate":
                    children = [children[step%len(children)]]
                for child in children:
                    propagate(child, output, step, node.weight)
            elif node:
                if output and output[-1][1] == weight:
                    output[-1][0] += str(node)
                else:
                    output.append([str(node), weight])
        output = []
        propagate(tree, output, step, 1.0)
        return output

    tree = prompt_grammar.parse(prompt)

    schedules = []
    for step in range(steps, 0, -1):
        scheduled = extract(tree, step)
        if not schedules or tuple(schedules[-1][1]) != tuple(scheduled):
            schedules += [(step, scheduled)]
    schedules = schedules[::-1]
    return schedules

def tokenize_prompt(clip, parsed):
    tokenizer = clip.tokenizer
    comma_token = tokenizer.comma_token_id

    # chunking sizes
    chunk_size = 75
    leeway = 20 

    # tokenize prompt and split it into chunks
    tokenized = tokenizer([text for text, _ in parsed])["input_ids"]
    tokenized = [tokens[1:-1] for tokens in tokenized] # strip special tokens

    # weight the individual tokens
    weighted = []
    for tokens, (_, weight) in zip(tokenized, parsed):
        weighted += [(t, weight) for t in tokens]
    tokenized = weighted

    # add TI embeddings inline with the tokens (our CLIP handles these separately)
    i = 0
    while i < len(tokenized):
        for name, vector in clip.textual_inversions:
            match = tuple([t for t,_ in tokenized[i:i+len(name)]])
            if match == name:
                weight = tokenized[i][1]
                tokenized = tokenized[:i] + [(v, weight) for v in vector] + tokenized[i+len(name):]
                i += vector.shape[0]
                break
        else:
            i += 1

    # split tokens into chunks
    chunks = []
    while tokenized:
        if len(tokenized) <= chunk_size:
            chunk = tokenized
        else:
            chunk = tokenized[:chunk_size]

            # split on a comma if its close to the end of the chunk
            commas = [i for i, (c, _) in enumerate(chunk) if c == comma_token and i > chunk_size - leeway]
            if commas:
                chunk = tokenized[:commas[-1]+1]

        tokenized = tokenized[len(chunk):]
        chunks += [chunk]
    
    if not chunks:
        chunks = [[]]
    
    # truncate chunks, only 3 allowed!
    #if len(chunks) > 3:
    #    chunks = chunks[:3]
    
    return chunks

def encode_tokens(clip, chunks, clip_skip=1):
    tokenizer = clip.tokenizer

    start_token = tokenizer.bos_token_id
    end_token = tokenizer.eos_token_id
    padding_token = tokenizer.pad_token_id

    chunk_encodings = []

    for chunk in chunks:
        # add special tokens and padding
        start = [(start_token, 1.0)]
        end = [(end_token, 1.0)]
        padding = [(padding_token, 1.0)] * (75-len(chunk))
        chunk = start + chunk + end + padding

        tokens, weights = list(zip(*chunk))

        # encode chunk tokens
        encoding = clip(tokens)

        # do clip skip
        encoding = encoding['hidden_states'][-clip_skip]
        encoding = clip.final_layer_norm(encoding)
        
        # each token has been encoded into its own tensor
        # we weight this tensor with the tokens weight
        weights = torch.tensor([weights], device=clip.device)
        weights = weights.reshape(weights.shape + (1,)).expand(encoding.shape)

        # keep the mean the same, lets the weighting operation work somewhat
        original_mean = encoding.mean()
        encoding = encoding * weights
        new_mean = encoding.mean()
        encoding = encoding * (original_mean / new_mean)

        chunk_encodings += [encoding]

    # combine all chunk encodings
    encoding = torch.hstack(chunk_encodings)
    return encoding

def get_encoding_at_step(schedule, step):
    for start, encoding in schedule:
        if start >= step:
            return encoding
    return schedule[-1][1]

class ConditioningSchedule():   
    def __init__(self, clip, prompt, negative_prompt, steps, clip_skip, batch_size):
        self.positives = []
        self.negatives = []

        tokenized_pos = []
        tokenized_neg = []

        for i in range(len(prompt)):
            schedule_pos = parse_prompt(prompt[i], steps)
            tokenized_pos += [[(steps, tokenize_prompt(clip, prompt)) for steps, prompt in schedule_pos]]

        for i in range(len(negative_prompt)):
            schedule_neg = parse_prompt(negative_prompt[i], steps)
            tokenized_neg += [[(steps, tokenize_prompt(clip, prompt)) for steps, prompt in schedule_neg]]

        max_chunks = max([max([len(chunk) for _, chunk in prompt]) for prompt in tokenized_pos + tokenized_neg])

        for i in range(len(tokenized_pos)):
            padded_tokenized_pos = [(steps, chunks + [chunks[-1]] * (max_chunks-len(chunks))) for steps, chunks in tokenized_pos[i]]
            encoded_pos = [(steps, encode_tokens(clip, chunks, clip_skip)) for steps, chunks in padded_tokenized_pos]
            self.positives += [encoded_pos]
        
        for i in range(len(tokenized_neg)):
            padded_tokenized_neg = [(steps, chunks + [chunks[-1]] * (max_chunks-len(chunks))) for steps, chunks in tokenized_neg[i]]
            encoded_neg = [(steps, encode_tokens(clip, chunks, clip_skip)) for steps, chunks in padded_tokenized_neg]
            self.negatives += [encoded_neg]

        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.offset = 0
    
    def __getitem__(self, step):
        step += self.offset

        conditioning_pos = []
        conditioning_neg = []

        for i in range(self.batch_size):
            p = i % len(self.positives)
            n = i % len(self.negatives)
            conditioning_pos += [get_encoding_at_step(self.positives[p], step)]
            conditioning_neg += [get_encoding_at_step(self.negatives[n], step)]

        return torch.cat(conditioning_neg + conditioning_pos)