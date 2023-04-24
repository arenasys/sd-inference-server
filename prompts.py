import lark
import re
import torch

class WeightedTree(lark.Tree):
    pass

prompt_grammar = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasis | deemphasis | numeric | scheduled | alternate | plain | addnet | WHITESPACE)*
emphasis: "(" prompt ")"
deemphasis: "[" prompt "]"
numeric: "(" prompt ":" [_WHITESPACE] NUMBER [_WHITESPACE]")"
scheduled: "[" [prompt ":"] prompt ":" [_WHITESPACE] specifier [_WHITESPACE]"]"
alternate: "[" prompt ("|" prompt)+ "]"
addnet: "<" [ local ] net_type ":" plain [ ":" [_WHITESPACE] specifier [_WHITESPACE] [ ":" [_WHITESPACE] specifier [_WHITESPACE] ]] ">"
net_type: LORA | HN
specifier: NUMBER | HR
local: "@"
HR: "HR"
LORA: "lora"
HN: "hypernet"
WHITESPACE: /\s+/
_WHITESPACE: /\s+/
plain: /([^\\\[\]()<>:|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""", tree_class=WeightedTree)

def parse_prompt(prompt, steps, HR=False):
    if not prompt:
        return [(steps, [["", 1.0]])]

    def extract(tree, step, HR=False):
        def propagate(node, output, step, HR, weight):
            if type(node) == WeightedTree:
                node.weight = weight
                children = node.children
                if node.data == "emphasis": node.weight *= 1.1
                if node.data == "deemphasis": node.weight /= 1.1
                if node.data == "numeric":
                    node.weight *= float(node.children[1])
                    children = [node.children[0]]
                if node.data == "scheduled":
                    specifier = node.children[2].children[0]
                    if specifier == "HR":
                        if not HR:
                            children = [node.children[0]]
                        else:
                            children = [node.children[1]]
                    else:
                        if step <= float(specifier):
                            children = [node.children[0]]
                        else:
                            children = [node.children[1]]
                if node.data == "alternate":
                    children = [children[step%len(children)]]
                if node.data == "addnet":
                    local = False
                    if children[0]:
                        local = True
                    children = children[1:]

                    name = str(children[0].children[0]) + ":" + str(children[1].children[0])

                    unet, clip = 1.0, None
                    if children[2]:
                        unet = float(children[2].children[0])
                    if children[3]:
                        clip = float(children[3].children[0])
                    if clip == None:
                        clip = unet

                    output.append((name, unet, clip, local))
                    children = []

                for child in children:
                    propagate(child, output, step, HR, node.weight)
            elif node:
                if output and type(output[-1]) == list and output[-1][1] == weight:
                    output[-1][0] += str(node)
                else:
                    output.append([str(node), weight])
        output = []
        propagate(tree, output, step, HR, 1.0)
        return output

    tree = prompt_grammar.parse(prompt)

    schedules = []
    for step in range(steps, 0, -1):
        scheduled = extract(tree, step, HR)
        if not schedules or tuple(schedules[-1][1]) != tuple(scheduled):
            schedules += [(step, scheduled)]
    schedules = schedules[::-1]
    return schedules

def tokenize_prompt(clip, parsed):
    tokenizer = clip.tokenizer
    comma_token = tokenizer.comma_token_id
    break_token = tokenizer.break_token_id

    # chunking sizes
    chunk_size = 75
    leeway = 20 

    # replace capitalized break with our break token
    text = [re.sub(r'BREAK[\.,\s]', '~~~ ', text) for text, _ in parsed]

    # tokenize prompt and split it into chunks
    if not text:
        text = ['']
    tokenized = tokenizer(text)["input_ids"]
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
        chunk = tokenized[:min(len(tokenized), chunk_size)]

        breaks = [i for i, (c, _) in enumerate(chunk) if type(c) == int and c == break_token]
        commas = [i for i, (c, _) in enumerate(chunk) if type(c) == int and c == comma_token and i > chunk_size - leeway]
        if breaks:
            # split on the first break and remove it from the prompt
            chunk = tokenized[:breaks[0]]
            del tokenized[breaks[0]]
        elif commas and len(tokenized) > chunk_size:
            # split on a comma if its close to the end of the chunk
            if commas:
                chunk = tokenized[:commas[-1]+1]

        tokenized = tokenized[len(chunk):]
        chunks += [chunk]
    
    if not chunks:
        chunks = [[]]
    
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
        encoding = clip.text_model(tokens)

        # do clip skip
        encoding = encoding['hidden_states'][-clip_skip]
        encoding = clip.text_model.final_layer_norm(encoding)
        
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

def seperate_schedule(schedule):
    prompt_schedule = []
    networks_schedule = []

    for i in range(len(schedule)):
        s = schedule[i][0]
        n = [t for t in schedule[i][1] if type(t) == tuple]
        t = [t for t in schedule[i][1] if type(t) == list]
        prompt_schedule += [(s,t)]
        networks_schedule += [(s,n)]
    
    return prompt_schedule, networks_schedule

class PromptSchedule():
    def __init__(self, parent, index, prompt, steps, clip, HR):
        self.parent = parent
        self.index = index
        self.schedule = parse_prompt(prompt, steps, HR)
        self.schedule, self.network_schedule = seperate_schedule(self.schedule)
        self.tokenized = [(steps, tokenize_prompt(clip, prompt)) for steps, prompt in self.schedule]
        self.chunks = max(len(p) for _, p in self.tokenized)
        self.encoded = None
        self.all_networks = {}
    
    def pad_to_length(self, max_chunks):
        self.tokenized = [(steps, chunks + [chunks[-1]] * (max_chunks-len(chunks))) for steps, chunks in self.tokenized]
        
    def encode(self, clip, clip_skip):
        clip_networks = [self.parent.get_networks_at_step(0,1)[self.index]]
        clip.additional.set_strength(clip_networks)
        self.encoded = [(steps, encode_tokens(clip, chunks, clip_skip)) for steps, chunks in self.tokenized]

    def get_encoding_at_step(self, step):
        for start, encoding in self.encoded:
            if start >= step:
                return encoding
        return encoding[-1][1]
    
    def get_all_networks(self):
        if self.all_networks:
            return self.all_networks
        all_networks = set()
        for _, networks in self.network_schedule:
            for net, _, _, _ in networks:
                all_networks.add(net)
        self.all_networks = all_networks
        return all_networks

    def get_networks_at_step(self, step):
        step_networks = {n:(0,0) for n in self.get_all_networks()}
        for start, network in self.network_schedule:
            if start >= step:
                for name, unet, clip, local in network:
                    step_networks[name] = (unet, clip, local)
                break
        
        return step_networks
    
class ConditioningSchedule():
    def __init__(self, clip, prompt, negative_prompt, steps, clip_skip, areas):
        self.clip = clip
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.clip_skip = clip_skip
        self.areas = areas
        self.HR = False
        self.parse()

    def switch_to_HR(self, hr_steps, areas):
        self.steps = hr_steps
        self.areas = areas
        self.HR = True
        self.parse()

    def parse(self):
        self.positives = [PromptSchedule(self, i, p, self.steps, self.clip, self.HR) for i, p in enumerate(self.prompt)]
        self.negatives = [PromptSchedule(self, i + len(self.positives), p, self.steps, self.clip, self.HR) for i, p in enumerate(self.negative_prompt)]

        max_chunks = max([p.chunks for p in self.positives + self.negatives])

        for p in self.positives + self.negatives:
            p.pad_to_length(max_chunks)

    def encode(self):
        for p in self.positives + self.negatives:
            p.encode(self.clip, self.clip_skip)

    def get_all_networks(self):
        networks = self.get_networks_at_step(0)
        all_networks = set()
        for n in networks:
            all_networks = all_networks.union(set(n.keys()))
        return all_networks
    
    def get_networks_at_step(self, step, idx=0):
        networks = [p.get_networks_at_step(step) for p in self.positives] + \
                   [n.get_networks_at_step(step) for n in self.negatives]
        local_networks = [{k:v[idx] for k,v in n.items() if v[-1]} for n in networks]

        global_networks = {}
        for network in networks:
            for k, v in network.items():
                if v[-1]:
                    continue
                global_networks[k] = max(global_networks.get(k, -10), v[idx])

        for k, v in global_networks.items():
            for network in local_networks:
                if not k in network:
                    network[k] = v
        
        return local_networks
    
    def get_conditioning_at_step(self, step):
        return [p.get_encoding_at_step(step) for p in self.positives] + \
               [n.get_encoding_at_step(step) for n in self.negatives]
    
    def get_composition(self, dtype, device):
        if self.areas:
            f = 0.8
            weights = [f] * len(self.positives)
            weights[0] = 1
            weights = torch.tensor(weights, dtype=dtype, device=device).reshape(-1,1,1,1)
            shape = self.areas[0].shape
            pos = [torch.ones(shape, dtype=dtype, device=device)] * len(self.positives)
            for i in range(len(self.areas)):
                if i + 1 < len(pos):
                    pos[i + 1] = self.areas[i].to(dtype=dtype, device=device)
            neg = torch.tensor([1] * len(self.negatives), dtype=dtype, device=device).reshape(-1,1,1,1)
            return [(torch.cat(pos), weights), neg]
        else:
            pos = torch.tensor([1] * len(self.positives), dtype=dtype, device=device).reshape(-1,1,1,1)
            neg = torch.tensor([1] * len(self.negatives), dtype=dtype, device=device).reshape(-1,1,1,1)
            return [(pos, pos), neg]
    
class BatchedConditioningSchedules():
    def __init__(self, clip, prompts, steps, clip_skip, areas):
        self.clip = clip
        self.prompts = prompts
        self.steps = steps
        self.clip_skip = clip_skip
        self.batch_size = len(prompts)
        self.areas = areas
        self.parse()

    def switch_to_HR(self, hr_steps, areas):
        for i, b in enumerate(self.batches):
            a = []
            if i < len(areas):
                a = areas[i]
            b.switch_to_HR(hr_steps, a)

    def parse(self):
        self.batches = []
        for i, (positive, negative) in enumerate(self.prompts):
            areas = []
            if i < len(self.areas):
                areas = self.areas[i]
            self.batches += [ConditioningSchedule(self.clip, positive, negative, self.steps, self.clip_skip, areas)]
    
    def encode(self):
        for b in self.batches:
            b.encode()        

    def get_all_networks(self):
        all_networks = set()
        for b in self.batches:
            all_networks = all_networks.union(b.get_all_networks())
        return all_networks
    
    def get_networks_at_step(self, step, idx=0):
        networks = []
        for b in self.batches:
            networks += b.get_networks_at_step(step, idx)
        return networks
    
    def get_conditioning_at_step(self, step):
        conditioning = []
        for b in self.batches:
            conditioning += b.get_conditioning_at_step(step)
        cond = torch.cat(conditioning)
        return cond
    
    def get_compositions(self, dtype, device):
        compositions = []
        for b in self.batches:
            compositions += [b.get_composition(dtype, device)]
        return compositions