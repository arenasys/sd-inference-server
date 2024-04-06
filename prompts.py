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
numeric: "(" prompt ":" [_WHITESPACE] strength_specifier [_WHITESPACE]")"
strength_specifier: NUMBER | strength_schedule
strength_schedule: "[" [strength_specifier ":"] strength_specifier ":" [_WHITESPACE] step_specifier [_WHITESPACE]"]"
scheduled: "[" [prompt ":"] prompt ":" [_WHITESPACE] step_specifier [_WHITESPACE]"]"
step_specifier: NUMBER | HR
alternate: "[" prompt ("|" prompt)+ "]"
addnet: "<" [ local ] net_type ":" filename [ ":" [_WHITESPACE] block_weight_specifier [_WHITESPACE] [ ":" [_WHITESPACE] strength_specifier [_WHITESPACE] ]] ">"
net_type: LORA
block_weight: [_WHITESPACE] strength_specifier [_WHITESPACE] "," [_WHITESPACE] strength_specifier [_WHITESPACE] ("," [_WHITESPACE] strength_specifier [_WHITESPACE])*
block_weight_specifier: NUMBER | block_weight | block_weight_schedule
block_weight_schedule: "[" [block_weight_specifier ":"] block_weight_specifier ":" [_WHITESPACE] step_specifier [_WHITESPACE]"]"
local: "@"
HR: "HR"
LORA: "lora"
WHITESPACE: /\s+/
_WHITESPACE: /\s+/
plain: /([^\\\[\]()<>:|]|\\.)+/
filename: /([^<>:]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""", tree_class=WeightedTree)

def parse_prompt(prompt, steps, HR=False):
    if not prompt:
        return [(steps, [["", 1.0]])]

    def extract(tree, step, total, HR=False):
        def triggered(specifier, step, total, HR):
            if specifier == "HR":
                return HR
            
            specifier = float(specifier)

            comparison = step
            if specifier < 1.0:
                comparison = step/total
            
            return specifier < comparison
        
        def resolve_strength(node, step, total, HR):
            strength = node.children[0]

            if type(strength) == lark.Token:
                if strength.type == "NUMBER":
                    return float(strength)
            else:
                if strength.data in {"strength_schedule", "block_weight_schedule"}:
                    specifier = strength.children[2].children[0]
                    active = strength.children[1 if triggered(specifier, step, total, HR) else 0]
                    return resolve_strength(active, step, total, HR)
                elif strength.data == "block_weight":
                    return [resolve_strength(c, step, total, HR) for c in strength.children]
                else:
                    print(strength)
                        
            return 1.0

        def propagate(node, output, step, total, HR, weight):
            if type(node) == WeightedTree:
                node.weight = weight
                children = node.children
                if node.data == "emphasis": node.weight *= 1.1
                if node.data == "deemphasis": node.weight /= 1.1
                if node.data == "numeric":
                    node.weight *= resolve_strength(node.children[1], step, total, HR)
                    children = [node.children[0]]
                if node.data == "scheduled":
                    specifier = node.children[2].children[0]
                    children = [node.children[1 if triggered(specifier, step, total, HR) else 0]]
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
                        unet = resolve_strength(children[2], step, total, HR)
                    if children[3]:
                        clip = resolve_strength(children[3], step, total, HR)
                    if clip == None:
                        if type(unet) == list:
                            clip = 1.0
                        else:
                            clip = unet

                    output.append((name, unet, clip, local))
                    children = []

                for child in children:
                    propagate(child, output, step, total, HR, node.weight)
            elif node:
                if output and type(output[-1]) == list and output[-1][1] == weight:
                    output[-1][0] += str(node)
                elif weight != 0.0:
                    output.append([str(node), weight])
        output = []
        propagate(tree, output, step, total, HR, 1.0)
        return output

    tree = prompt_grammar.parse(prompt)

    schedules = []
    for step in range(steps, 0, -1):
        scheduled = extract(tree, step, steps, HR)
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

    def replace_keyword(k, v, t):
        t = re.sub(fr'{k}[,\.]?[^\S\r\n]?', f'{v} ', t)
        return t

    # replace BREAK with our break token
    text = [replace_keyword("BREAK", "~~~", t) for t, _ in parsed]

    # replace START and END keywords with their tokens
    text = [replace_keyword("START", "<|startoftext|>", t) for t in text]
    text = [replace_keyword("END", "<|endoftext|>", t) for t in text]

    # remove escape backslashes
    text = [re.sub(r'\\(.)', r'\g<1>', t) for t in text]

    if not text:
        text = ['']

    # tokenize prompt and split it into chunks
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
    chunk_inversions = []
    pooled_text_embs = []

    for chunk in chunks:
        # add special tokens and padding
        start = [(start_token, 1.0)]
        end = [(end_token, 1.0)]
        padding = [(padding_token, 1.0)] * (75-len(chunk))
        chunk = start + chunk + end + padding

        tokens, weights = list(zip(*chunk))

        # encode chunk tokens
        encoding, pooled_text_emb = clip.encode(tokens, clip_skip)
        
        # each token has been encoded into its own tensor
        # we weight this tensor with the tokens weight
        
        inverted = [i for i in range(len(weights)) if weights[i] < 0]
        weights = torch.tensor([abs(w) for w in weights], device=clip.device)
        weights = weights.reshape(weights.shape + (1,)).expand(encoding.shape)
        

        # keep the mean the same, lets the weighting operation work somewhat
        # dont normalize small means to avoid fp issues
        original_mean = encoding.mean()
        if abs(original_mean.item()) > 0.01:
            encoding = encoding * weights
            new_mean = encoding.mean()
            encoding = encoding * (original_mean / new_mean)
        else:
            encoding = encoding * weights

        chunk_encodings += [encoding]
        pooled_text_embs += [pooled_text_emb]
        chunk_inversions += [inverted]

    # combine all chunk encodings
    encoding = torch.hstack(chunk_encodings)
    if all([p != None for p in pooled_text_embs]):
        pooled_text_emb = pooled_text_embs[0]
    else:
        pooled_text_emb = None
    
    inversions = []
    for c in range(len(chunk_inversions)):
        for i in chunk_inversions[c]:
            inversions += [int(c*77 + i)]
    
    return encoding, pooled_text_emb, inversions

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
    def __init__(self, parent, index, prompt, steps, HR):
        self.parent = parent
        self.index = index
        self.weight = None
        prompt, self.weight = self.get_weight(prompt)
        self.schedule = parse_prompt(prompt, steps, HR)
        self.schedule, self.network_schedule = seperate_schedule(self.schedule)
        self.encoded = None
        self.pooled_text_embs = None
        self.all_networks = {}

    def get_weight(self, prompt):
        parts = prompt.rsplit(":", 1)
        if len(parts) < 2:
            return prompt, None
        try:
            return parts[0].strip(), float(parts[-1].strip())
        except:
            return prompt, None
    
    def pad_to_length(self, max_chunks):
        self.tokenized = [(steps, chunks + [chunks[-1]] * (max_chunks-len(chunks))) for steps, chunks in self.tokenized]
        
    def tokenize(self, clip):
        self.tokenized = [(steps, tokenize_prompt(clip, prompt)) for steps, prompt in self.schedule]
        self.chunks = max(len(p) for _, p in self.tokenized)

    def encode(self, clip, clip_skip):
        clip_networks = [self.parent.get_networks_at_step(0,1)[self.index]]
        clip.additional.set_strength(clip_networks)
        self.encoded = [(steps, *encode_tokens(clip, chunks, clip_skip)) for steps, chunks in self.tokenized]

    def get_encoding_at_step(self, step):
        for start, encoding, _, _ in self.encoded:
            if start >= step:
                return encoding
        return encoding
    
    def get_pooled_text_embed_at_step(self, step):
        for start, _, pooled_text_embed, _ in self.encoded:
            if start >= step:
                return pooled_text_embed
        return pooled_text_embed
    
    def get_inversions_at_step(self, step):
        for start, _, _, inversions in self.encoded:
            if start >= step:
                return inversions
        return inversions
    
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
    def __init__(self, prompt, negative_prompt, steps, clip_skip):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.clip_skip = clip_skip
        self.areas = None
        self.HR = False
        self.model_type = None
        self.parse()

    def switch_to_HR(self, hr, steps):
        self.steps = steps
        self.HR = hr
        self.parse()

    def parse(self):
        self.positives = [PromptSchedule(self, i, p, self.steps, self.HR) for i, p in enumerate(self.prompt)]
        self.negatives = [PromptSchedule(self, i + len(self.positives), p, self.steps, self.HR) for i, p in enumerate(self.negative_prompt)]

    def tokenize(self, clip):
        for p in self.positives + self.negatives:
            p.tokenize(clip)

    def max_chunks(self):
        return max([p.chunks for p in self.positives + self.negatives])

    def pad_to_length(self, max_chunks):
        for p in self.positives + self.negatives:
            p.pad_to_length(max_chunks)

    def encode(self, clip, areas):
        self.areas = areas
        self.model_type = clip.model_type

        for p in self.positives + self.negatives:
            p.encode(clip, self.clip_skip)

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

                a = v[idx]
                b = global_networks.get(k, -10)

                if type(a) == float and type(b) == float:
                    global_networks[k] = max(a, b)
                    continue

                a_avg = sum(a)/len(a) if type(a) == list else a
                b_avg = sum(b)/len(b) if type(b) == list else b
                global_networks[k] = a if a_avg > b_avg else b

        for k, v in global_networks.items():
            for network in local_networks:
                if not k in network:
                    network[k] = v
        
        return local_networks
    
    def get_conditioning_at_step(self, step):
        return [p.get_encoding_at_step(step) for p in self.positives] + \
               [n.get_encoding_at_step(step) for n in self.negatives]
    
    def get_additional_conditioning_at_step(self, step):
        out = {}

        if self.model_type == "SDXL-Base":
            text_embeds = [p.get_pooled_text_embed_at_step(step) for p in self.positives] + \
                        [n.get_pooled_text_embed_at_step(step) for n in self.negatives]
            z = 1024
            time_ids = [torch.tensor([z, z, 0, 0, z, z]) for _ in self.positives + self.negatives]
            out["text_embeds"] = text_embeds
            out["time_ids"] = time_ids
        
        return out

    def get_additional_attention_kwargs_at_step(self, step):    
        out = {}
        inversions = [p.get_inversions_at_step(step) for p in self.positives] + \
                     [n.get_inversions_at_step(step) for n in self.negatives]
        out["token_inversions"] = inversions
        return out
    
    def get_composition(self, dtype, device):
        pos_w = [p.weight for p in self.positives]
        pos_w[0] = pos_w[0] or 1
        pos_w = [w or 0.8 for w in pos_w]

        neg_w = [n.weight or 1.0 for n in self.negatives]

        if self.areas:
            weights = torch.tensor(pos_w, dtype=dtype, device=device).reshape(-1,1,1,1)
            shape = self.areas[0].shape
            pos = [torch.ones(shape, dtype=dtype, device=device)] * len(self.positives)
            for i in range(len(self.areas)):
                if i + 1 < len(pos):
                    pos[i + 1] = self.areas[i].to(dtype=dtype, device=device)
            neg = torch.tensor(neg_w, dtype=dtype, device=device).reshape(-1,1,1,1)
            return [(torch.cat(pos), weights), neg]
        else:
            pos = torch.tensor(pos_w, dtype=dtype, device=device).reshape(-1,1,1,1)
            neg = torch.tensor(neg_w, dtype=dtype, device=device).reshape(-1,1,1,1)
            mask = torch.tensor([1] * len(self.positives), dtype=dtype, device=device).reshape(-1,1,1,1)
            return [(pos, mask), neg]
    
class BatchedConditioningSchedules():
    def __init__(self, prompts, steps, clip_skip):
        self.prompts = prompts
        self.steps = steps
        self.clip_skip = clip_skip
        self.batch_size = len(prompts)
        self.parse()

    def switch_to_HR(self, hr_steps):
        for i, b in enumerate(self.batches):
            b.switch_to_HR(True, hr_steps)

    def parse(self):
        self.batches = []
        for i, (positive, negative) in enumerate(self.prompts):
            self.batches += [ConditioningSchedule(positive, negative, self.steps, self.clip_skip)]
    
    def encode(self, clip, areas):
        max_chunks = 0
        for b in self.batches:
            b.tokenize(clip)
            max_chunks = max(max_chunks, b.max_chunks())
        
        for b in self.batches:
            b.pad_to_length(max_chunks)
        
        for i, b in enumerate(self.batches):
            a = areas[i] if i < len(areas) else []
            b.encode(clip, a)
    
    def get_all_networks(self, hr_steps=None):
        current_networks = set()
        hr_networks = set()

        for b in self.batches:
            current_networks = current_networks.union(b.get_all_networks())
            if hr_steps != None:
                b.switch_to_HR(True, hr_steps)
                hr_networks = hr_networks.union(b.get_all_networks())
                b.switch_to_HR(False, self.steps)

        if hr_steps != None:
            return current_networks, hr_networks.union(current_networks)

        return current_networks, current_networks
    
    def get_networks_at_step(self, step, idx=0):
        networks = []
        for b in self.batches:
            networks += b.get_networks_at_step(step, idx)
        return networks
    
    def get_initial_networks(self):
        unets = self.get_networks_at_step(0,0)
        clips = self.get_networks_at_step(0,1)
        return unets[0], clips[0]
    
    def get_conditioning_at_step(self, step, dtype, device):
        conditioning = []
        for b in self.batches:
            conditioning += b.get_conditioning_at_step(step)
        cond = torch.cat(conditioning).to(device, dtype)
        return cond
    
    def get_additional_conditioning_at_step(self, step, dtype, device):
        add_conditioning = {}

        for b in self.batches:
            add_cond = b.get_additional_conditioning_at_step(step)
            for k in add_cond:
                if not k in add_conditioning:
                    add_conditioning[k] = []
                add_conditioning[k] += add_cond[k]

        for k in add_conditioning:
            add_conditioning[k] = torch.stack(add_conditioning[k]).to(device, dtype)

        return add_conditioning
    
    def get_additional_attention_kwargs_at_step(self, step):
        add_kwargs = {}

        for b in self.batches:
            b_add_kwargs = b.get_additional_attention_kwargs_at_step(step)
            for k, v in b_add_kwargs.items():
                if not k in add_kwargs:
                    add_kwargs[k] = []
                add_kwargs[k] += v
        
        for k in list(add_kwargs.keys()):
            if not any(add_kwargs[k]):
                del add_kwargs[k]

        return add_kwargs
    
    def get_compositions(self, dtype, device):
        compositions = []
        for b in self.batches:
            compositions += [b.get_composition(dtype, device)]
        return compositions