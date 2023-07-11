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
        weights = torch.tensor([weights], device=clip.device)
        weights = weights.reshape(weights.shape + (1,)).expand(encoding.shape)

        # keep the mean the same, lets the weighting operation work somewhat
        original_mean = encoding.mean()
        encoding = encoding * weights
        new_mean = encoding.mean()
        encoding = encoding * (original_mean / new_mean)

        chunk_encodings += [encoding]
        pooled_text_embs += [pooled_text_emb]

    # combine all chunk encodings
    encoding = torch.hstack(chunk_encodings)
    if all([p != None for p in pooled_text_embs]):
        pooled_text_emb = torch.hstack(pooled_text_embs)
    else:
        pooled_text_emb = None
    
    return encoding, pooled_text_emb

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
        for start, encoding, _ in self.encoded:
            if start >= step:
                return encoding
        return encoding
    
    def get_pooled_text_embed_at_step(self, step):
        for start, _, pooled_text_embed in self.encoded:
            if start >= step:
                return pooled_text_embed
        return pooled_text_embed
    
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

    def switch_to_HR(self, hr_steps):
        self.steps = hr_steps
        self.HR = True
        self.parse()

    def parse(self):
        self.positives = [PromptSchedule(self, i, p, self.steps, self.HR) for i, p in enumerate(self.prompt)]
        self.negatives = [PromptSchedule(self, i + len(self.positives), p, self.steps, self.HR) for i, p in enumerate(self.negative_prompt)]

    def encode(self, clip, areas):
        self.areas = areas
        self.model_type = clip.model_type

        for p in self.positives + self.negatives:
            p.tokenize(clip)

        max_chunks = max([p.chunks for p in self.positives + self.negatives])

        for p in self.positives + self.negatives:
            p.pad_to_length(max_chunks)
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
                global_networks[k] = max(global_networks.get(k, -10), v[idx])

        for k, v in global_networks.items():
            for network in local_networks:
                if not k in network:
                    network[k] = v
        
        return local_networks
    
    def get_conditioning_at_step(self, step):
        return [p.get_encoding_at_step(step) for p in self.positives] + \
               [n.get_encoding_at_step(step) for n in self.negatives]
    
    def get_additional_conditioning_at_step(self, step):
        if self.model_type == "SDXL-Base":
            text_embeds = [p.get_pooled_text_embed_at_step(step) for p in self.positives] + \
                        [n.get_pooled_text_embed_at_step(step) for n in self.negatives]
            time_ids = [torch.tensor([1024, 1024, 0,0, 1024, 1024]) for _ in self.positives + self.negatives]
            return {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            return {}
    
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
            b.switch_to_HR(hr_steps)

    def parse(self):
        self.batches = []
        for i, (positive, negative) in enumerate(self.prompts):
            self.batches += [ConditioningSchedule(positive, negative, self.steps, self.clip_skip)]
    
    def encode(self, clip, areas):
        for i, b in enumerate(self.batches):
            a = areas[i] if i < len(areas) else []
            b.encode(clip, a)

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
    
    def get_initial_networks(self, comparable=False):
        unets = self.get_networks_at_step(0,0)
        clips = self.get_networks_at_step(0,1)

        all = set()
        for d in unets + clips:
            all = all.union(set(d.keys()))
        all = list(all)

        unet = unets[0]
        clip = clips[0]

        if comparable:
            unet_t = tuple([tuple([k,unet[k]]) for k in sorted(unet.keys())])
            clip_t = tuple([tuple([k,clip[k]]) for k in sorted(clip.keys())])
            all_t = tuple(sorted(all))
            return tuple([all_t, unet_t, clip_t])
        else:
            return all, unet, clip
    
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
    
    def get_compositions(self, dtype, device):
        compositions = []
        for b in self.batches:
            compositions += [b.get_composition(dtype, device)]
        return compositions