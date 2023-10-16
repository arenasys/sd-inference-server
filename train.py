import torch
import torch.utils.data
import glob
import os
import math
import PIL.Image
import tqdm
import torchvision.transforms as transforms
import random

import utils
import upscalers
import lora
import prompts

@torch.inference_mode()
def encode_images(vae, batch):
    seeds = [random.randrange(2147483646) for _ in batch]
    latents = utils.encode_images(vae, seeds, batch).to(torch.device("cpu"))
    return latents

@torch.inference_mode()
def tokenize_prompts(clip, batch):
    tokenized_weighted = [prompts.tokenize_prompt(clip, [[prompt, 1.0]]) for prompt in batch]
    tokenized = [[[token for token, weight in chunk] for chunk in prompt] for prompt in tokenized_weighted]
    max_chunks = max([len(chunks) for chunks in tokenized])
    padded = [chunks + [[]] * (max_chunks-len(chunks)) for chunks in tokenized]

    start = [clip.tokenizer.bos_token_id]
    end = [clip.tokenizer.eos_token_id]
    padding = [clip.tokenizer.pad_token_id]

    tokens = [[start + chunk + end + padding * (75-len(chunk)) for chunk in prompt] for prompt in padded]
    return tokens

def encode_tokens(clip, batch, clip_skip):
    batch_encodings = []
    for chunks in batch:
        chunk_encodings = []
        for chunk in chunks:
            encoding, _ = clip.encode(chunk, clip_skip)
            chunk_encodings += [encoding]
        batch_encodings += [torch.hstack(chunk_encodings)]
    return batch_encodings

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LoRAForward(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.lora_module = None
        self.original_module = original
        self.original_forward = original.forward

    def forward(self, x, *args, **kwargs):
        if self.lora_module:
            return self.original_forward(x) + self.lora_module(x)
        else:
            return self.original_forward(x)

def hijack_model(model, prefix, targets):
    modules = {}
    for module_name, module in model.named_modules():
        if not module.__class__.__name__ in targets:
            continue
        if "LoRA" in module.__class__.__name__:
            name = (prefix + '.' + module_name).replace(".", "_")
            if name in modules:
                continue
            modules[name] = LoRAForward(module)
            module.forward = modules[name].forward
        else:
            for child_name, child_module in module.named_modules():
                child_class = child_module.__class__.__name__
                if not child_name:
                    continue
                if child_class == "Linear" or child_class == "Conv2d":
                    name = (prefix + '.' + module_name + '.' + child_name).replace(".", "_")
                    if name in modules:
                        continue
                    modules[name] = LoRAForward(child_module)
                    child_module.forward = modules[name].forward
    return modules

def build_lora(rank, conv_rank, alpha, unet, clip):
    modules = hijack_model(unet, 'unet', ["LoRACompatibleLinear", "LoRACompatibleConv", "Transformer2DModel", "Attention", "ResnetBlock2D", "Downsample2D", "Upsample2D"])
    modules.update(hijack_model(clip, 'te', ["CLIPAttention", "CLIPMLP"]))
    network = lora.LoRANetwork()

    used_modules = {}
    for name, module in modules.items():
        is_conv = "resnet" in name or "sample" in name
        if is_conv and not conv_rank:
            continue
        dim = conv_rank if is_conv else rank
        lora_module = lora.LoRAModule.from_module(network.net_name, name, module.original_module, dim, alpha)
        network.add_module(name, lora_module)
        module.lora_module = lora_module
        used_modules[name] = module

    return network, used_modules

def get_state_dict(network):
    state_dict = network.state_dict()
    for k in list(state_dict.keys()):
        kk = k.replace("te_model", "lora_te").replace("unet", "lora_unet")
        if k != kk:
            state_dict[kk] = state_dict[k]
            del state_dict[k]
    for k in list(state_dict.keys()):
        state_dict[k] = state_dict[k].to(torch.device("cpu"), torch.float16)
    return state_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.clear()

    def configure(self, vae, clip, batch_size, image_size):
        self.vae = vae
        self.clip = clip

        self.batch_size = batch_size
        self.image_size = image_size

    def clear(self):
        self.original = {}
        self.cache = {}
        self.found = []
        self.epoch = []

    def add_path(self, path):
        found = []
        for e in ["jpg", "jpeg", "png"]:
            for img in glob.glob(os.path.join(path, "*."+e)):
                with PIL.Image.open(img) as i:
                    size = i.size
                found += [(img, size)]

        self.found += found

    def add_image(self, image, prompt):
        name = f"MEMORY-{len(self.original)}"
        self.original[name] = (image, prompt)
        self.found += [(name, image.size)]

    def find_best_bucket(self, size, buckets):
        zw, zh = size
        min_err = None
        bucket = None
        scaled = None
        for (bw, bh) in buckets:
            if zw < bw:
                f = bw / zw
            else:
                f = bh / zh
            (w, h) = (int(zw * f), int(zh * f))
            err = abs((zw/zh) - (bw/bh))
            if min_err == None or err < min_err:
                min_err = err
                bucket = (bw, bh)
                scaled = (w, h)
        return bucket, scaled

    def calculate_buckets(self):
        sizes = {}

        area = self.image_size * self.image_size
        for _, (w,h) in self.found:
            factor = (area / (w*h))**0.5
            w, h = int(w * factor), int(h * factor)
            w, h = w - w % 8, h - h % 8
            z = (w,h)
            if not z in sizes:
                sizes[z] = 0
            sizes[z] += 1

        buckets = set()

        #for z, c in list(sizes.items()):
        #    if c >= self.batch_size:
        #        buckets.add(z)

        for a in [1, 1.1, 1.25, 1.33, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.5, 5, 5.5, 6, 6.25]:
            w = int((area / a) ** 0.5)
            w = w - w % 8
            h = int(area / w)
            h = h - h % 8
            buckets.add((w,h))
            if w != h:
                buckets.add((h,w))

        while True:
            matches = {b:[] for b in buckets}

            for z, c in sizes.items():
                b, _ = self.find_best_bucket(z, buckets)
                matches[b] += [z] * c

            for z, m in list(matches.items()):
                if len(m) < self.batch_size:
                    buckets.discard(z)
            
            if len(buckets) == len(matches):
                break
        
        self.buckets = buckets

    def crop_to_bucket(self, image):
        bucket, _ = self.find_best_bucket(image.size, self.buckets)
        image = upscalers.upscale_single(image, transforms.InterpolationMode.LANCZOS, *bucket)
        return image
    
    def get_text_file(self, file):
        potential = [
            file+".txt",
            file.rsplit(".",1)[0]+".txt"
        ]
        for p in potential:
            if os.path.exists(p):
                return p
        return None
    
    def process_prompt(self, prompt):
        tags = [p.strip() for p in prompt.split(",")]
        random.shuffle(tags)
        return ", ".join(tags)
    
    def get_latent_and_prompt(self, file):
        if not file in self.cache:
            if file in self.original:
                img, text = self.original[file]
            else:
                img = PIL.Image.open(file).convert("RGB")
                text_file, text = self.get_text_file(file), ""
                if text_file:
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read()
            img = self.crop_to_bucket(img)
            latent = encode_images(self.vae, [img])[0]
            self.cache[file] = (latent, text)

        file, text = self.cache[file]
        
        return file, text

    def build_cache(self, callback=None):
        self.cache = {}
        iter = tqdm.tqdm(self.found, "CACHING")
        for file, _ in iter:
            if callback:
                callback(iter.format_dict)
            self.get_latent_and_prompt(file)

    def calculate_batches(self):
        bucketed = {}
        for file, size in self.found:
            bucket, _ = self.find_best_bucket(size, self.buckets)
            if not bucket in bucketed:
                bucketed[bucket] = []
            bucketed[bucket] += [file]

        batches = []
        for z in list(bucketed.keys()):
            files = [i for i in bucketed[z]]
            random.shuffle(files)
            batch_count = math.ceil(len(files)/self.batch_size)
            for i in range(batch_count):
                batch = []
                for j in range(self.batch_size):
                    idx = (i * batch_count + j) % len(files)
                    batch += [files[idx]]
                batches += [batch]
        random.shuffle(batches)

        self.epoch = batches

    def __getitem__(self, index):
        batch = self.epoch[index]

        data = [self.get_latent_and_prompt(file) for file in batch]

        batch_prompts = [self.process_prompt(prompt) for _, prompt in data]
        tokenized = tokenize_prompts(self.clip, batch_prompts)

        batch_latents = [latent for latent, _ in data]
        latents = torch.stack(batch_latents)

        return index, latents, tokenized
    
    def __len__(self):
        return len(self.epoch)
    
    def total_steps(self, epochs):
        return epochs * len(self.epoch)
    
    def total_epochs(self, steps):
        return math.ceil(steps / len(self.epoch))