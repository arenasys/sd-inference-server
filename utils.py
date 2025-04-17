import PIL
import PIL.Image
import PIL.ImageFilter
import PIL.ImageChops

import torch
import torchvision.transforms as transforms
import requests
import tqdm
import os
import time
import pickle
import re
import math

import numpy as np

DIRECTML_AVAILABLE = False
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except:
    pass

TO_TENSOR = transforms.ToTensor()
FROM_TENSOR = transforms.ToPILImage()

MAPPINGS = {}

def preprocess_images(images):
    def process(image):
        image = TO_TENSOR(image).to(torch.float32)
        return 2.0 * image[None, :] - 1.0
    return torch.cat([process(i) for i in images])

def postprocess_images(images):
    def process(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        return FROM_TENSOR(image)
    return [process(i) for i in images]

def blur_areas(areas, blur, expand):
    if not areas:
        return []
    out = []
    for i in range(len(areas)):
        o = []
        for j in range(len(areas[i])):
            a = areas[i][j]
            if expand or blur:
                b = a.copy()
                if expand:
                    b = b.filter(PIL.ImageFilter.MaxFilter(int(2*expand + 1)))
                if blur:
                    b = b.filter(PIL.ImageFilter.MaxFilter(int(blur*2 + 1)))
                    b = b.filter(PIL.ImageFilter.GaussianBlur(blur))
                a = PIL.ImageChops.lighter(a, b)
            o += [a]
        out += [o]
    return out

def preprocess_areas(areas, width, height):
    if not areas:
        return []
    out = []
    for i in range(len(areas)):
        o = []
        for j in range(len(areas[i])):
            w, h = width, height
            w, h = w - w % 8, h - h % 8
            a = areas[i][j].resize((w // 8, h // 8), resample=PIL.Image.LANCZOS)
            a = TO_TENSOR(a).to(torch.float32)
            o += [torch.cat([a]*4)[None, :]]
        out += [o]
    return out

def preprocess_masks(masks):
    def process(mask):
        w, h = mask.size
        w, h = w - w % 8, h - h % 8
        mask = mask.resize((w // 8, h // 8), resample=PIL.Image.LANCZOS)
        mask = 1 - TO_TENSOR(mask).to(torch.float32)
        return mask[None, :]
    return torch.cat([process(m) for m in masks])

def encode_inpainting(images, masks, vae, seeds):
    if masks != None:
        if type(masks) != torch.Tensor:
            masks = torch.cat([TO_TENSOR(m)[None,:] for m in masks])
        masks = masks.to(vae.device, vae.dtype)
        masks = torch.round(masks)

        if type(images) != torch.Tensor:
            images = torch.cat([TO_TENSOR(i)[None,:] for i in images])
        images = images.to(vae.device, vae.dtype)
        images = 2.0 * images - 1.0
        images = images * (1.0 - masks)
    else:
        masks_shape = (images.shape[0], 1, images.shape[2], images.shape[3])
        masks = torch.ones(masks_shape).to(vae.device, vae.dtype)
        images = torch.zeros(images.shape).to(vae.device, vae.dtype)

    latents = encode_images(vae, seeds, images)
    masks = torch.nn.functional.interpolate(masks, size=(masks.shape[2]//8,masks.shape[3]//8))

    return latents, masks

def encode_images(vae, seeds, images):
    if type(images) != torch.Tensor:
        images = preprocess_images(images).to(vae.device, vae.dtype)
    
    noise = singular_noise(seeds, images.shape[3] // 8, images.shape[2] // 8, vae.device).to(vae.dtype)

    dists = vae.encode(images)
    mean = torch.stack([dists.mean[i%len(images)] for i in range(len(seeds))])
    std = torch.stack([dists.std[i%len(images)] for i in range(len(seeds))])

    latents = (mean + std * noise) * vae.scaling_factor
    
    return latents

def encode_images_disjointed(vae, seeds, images):
    latents = []
    for i, seed in enumerate(seeds):
        latents += [encode_images(vae, [seed], [images[i%len(images)]])]
    return latents
    
def decode_images(vae, latents):
    latents = latents.clone().detach().to(vae.device, vae.dtype) / vae.scaling_factor
    images = vae.decode(latents).sample
    return postprocess_images(images)

def get_latents(vae, seeds, images):
    if type(images) == torch.Tensor:
        return images.to(vae.device)
    elif type(images) == list:
        if all([images[0].size == i.size for i in images]):
            return encode_images(vae, seeds, images)
        else:
            return encode_images_disjointed(vae, seeds, images)

def get_masks(device, masks):
    if type(masks) == torch.Tensor:
        return masks      
    elif type(masks) == list:
        return preprocess_masks(masks).to(device)
    
def prepare_mask(mask, blur, expand):
    if expand:
        mask = mask.filter(PIL.ImageFilter.MaxFilter(int(2*expand + 1)))
    if blur:
        mask = mask.filter(PIL.ImageFilter.GaussianBlur(blur))
    return mask

def apply_inpainting(images, originals, masks, extents):
    outputs = [None] * len(images)
    masked = [None] * len(images)
    for i in range(len(images)):
        image = images[i]
        original = originals[i%len(originals)]
        mask = masks[i%len(masks)]
        extent = extents[i%len(extents)]
        x1,y1,x2,y2 = extent

        image = image.convert("RGBA")
        mask = mask.convert("L")

        masked[i] = image.copy()
        masked[i].putalpha(mask.resize(image.size))

        ew, eh = x2-x1, y2-y1
        if (ew, eh) != images[i].size:
            image = image.resize((ew, eh))
            mask = mask.resize((ew, eh))

        outputs[i] = original.copy()
        outputs[i].paste(image, extent, mask)
    return outputs, masked

def get_extents(originals, masks, padding, width, height):
    def pad_extent(extent, p, src, wrk):
        wrk_w, wrk_h = wrk
        src_w, src_h = src

        if not extent:
            return 0, 0, src_w, src_h

        x1, y1, x2, y2 = extent

        ar = wrk_w/wrk_h
        cx,cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
        rw,rh = min(src_w, (x2-x1)+p), min(src_h, (y2-y1)+p)

        if wrk_w/rw < wrk_h/rh:
            w = rw
            h = int(w/ar)
            if h > src_h:
                h = src_h
                w = int(h*ar)
        else:
            h = rh
            w = int(h*ar)
            if w > src_w:
                w = src_w
                h = int(w/ar)

        x1 = cx - w//2
        x2 = cx + w - (w//2)

        if x1 < 0:
            x2 += -x1
            x1 = 0
        if x2 > src_w:
            x1 -= x2-src_w
            x2 = src_w

        y1 = cy - h//2
        y2 = cy + h - (h//2)

        if y1 < 0:
            y2 += -y1
            y1 = 0
        if y2 > src_h:
            y1 -= y2-src_h
            y2 = src_h

        return int(x1), int(y1), int(x2), int(y2)
    
    if not masks:
        return [(0,0,o.size[0],o.size[1]) for o in originals]
    
    if padding == None:
        padding = 10240

    extents = []
    for i in range(len(masks)):
        if masks[i] != None:
            extents += [pad_extent(masks[i].getbbox(), padding, originals[i].size, (width, height))]
        else:
            extents += [(0,0,originals[i].size[0],originals[i].size[1])]

    return extents

def apply_extents(inputs, extents):
    outputs = [None for i in inputs]
    for i in range(len(inputs)):
        extent = extents[i]
        outputs[i] = inputs[i].crop(extent)
    return outputs

def cast_state_dict(state_dict, dtype, device='cpu'):
    for k in state_dict:
        if type(state_dict[k]) == torch.Tensor and state_dict[k].dtype != dtype and state_dict[k].dtype in {torch.float16, torch.float32, torch.bfloat16}:
            tmp = state_dict[k].clone().detach()
            state_dict[k] = tmp.to(device, dtype=dtype)
    return state_dict

class NoiseSchedule():
    def __init__(self, seeds, subseeds, width, height, device, dtype):
        self.seeds = seeds
        self.subseeds = subseeds
        self.shape = (4, int(height), int(width))
        self.device = device
        self.dtype = dtype
        self.steps = 0
        self.index = 0

        self.noise = []

        self.reset()
    
    def reset(self):
        self.noise = []
        self.index = 0
        self.allocate(32)
    
    def allocate(self, steps):
        self.steps = steps
        self.generate()

    def generate(self):
        rng_device = self.device
        if DIRECTML_AVAILABLE and rng_device == torch_directml.device():
            rng_device = torch.device("cpu")

        generator = torch.Generator(rng_device)
        noises = []
        for i in range(len(self.seeds)):
            seed = self.seeds[i]
            generator.manual_seed(seed)
            noises += [[]]

            for _ in range(self.steps+1):
                noise = torch.randn(self.shape, generator=generator, device=rng_device).to(self.device, self.dtype)
                noises[i] += [noise]

        for i in range(len(self.subseeds)):
            seed, strength = self.subseeds[i]
            generator.manual_seed(seed)
            subnoise = torch.randn(self.shape, generator=generator, device=rng_device).to(self.device, self.dtype)
            noises[i][0] = slerp_noise(strength, noises[i][0], subnoise)

        self.noise = [torch.stack(n) for n in zip(*noises)]
    
    def __getitem__(self, i):
        if i >= len(self.noise):
            self.steps *= 2
            self.generate()
        return self.noise[i]

    def __call__(self, advance=True):
        noise = self[self.index]
        if advance:
            self.index += 1
        return noise

def singular_noise(seeds, width, height, device):
    rng_device = device
    if DIRECTML_AVAILABLE and rng_device == torch_directml.device():
        rng_device = torch.device("cpu")

    shape = (4, int(height), int(width))
    generator = torch.Generator(rng_device)
    noises = []
    for i in range(len(seeds)):
        generator.manual_seed(seeds[i])
        noises += [[]]
        noises[i] = torch.randn(shape, generator=generator, device=rng_device).to(device)
        
    combined = torch.stack(noises)
    return combined

def slerp_noise(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1)*high
    return res

class DisableInitialization:
    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        self.init_kaiming_uniform = torch.nn.init.kaiming_uniform_
        self.init_no_grad_normal = torch.nn.init._no_grad_normal_

        torch.nn.init.kaiming_uniform_ = do_nothing
        torch.nn.init._no_grad_normal_ = do_nothing

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.init.kaiming_uniform_ = self.init_kaiming_uniform
        torch.nn.init._no_grad_normal_ = self.init_no_grad_normal

class CUDATimer:
    def __init__(self, label):
        self.label = label
    
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        print(self.label, self.start.elapsed_time(self.end))

def download(url, path, progress_callback=None, started_callback=None, headers={}):
    if os.path.isdir(path):
        filename = None
        folder = path
    else:
        filename = path
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=10, headers=headers, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))

    content_length = resp.headers.get("content-length", 0)
    if not content_length:
        raise RuntimeError(f"response is empty")

    content_type = resp.headers.get("content-type", "unknown")
    content_disposition = resp.headers.get("content-disposition", "")

    if not content_type in {"application/zip", "binary/octet-stream", "application/octet-stream", "multipart/form-data"}:
        if not (content_type == "unknown" and "attachment" in content_disposition):
            raise RuntimeError(f"{content_type} content type is not supported")

    if not filename:
        if content_disposition:
            filename = re.findall("filename=\"(.+)\";?", content_disposition)[0]
        else:
            filename = url.rsplit("/",-1)[-1]
        filename = os.path.join(folder, filename)

    if started_callback:
        started_callback(filename, resp)

    desc = filename.rsplit(os.path.sep)[-1]

    last = None
    with open(filename+".tmp", 'wb') as file, tqdm.tqdm(desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            if not last or time.time() - last > 0.5:
                last = time.time()
                if progress_callback:
                    progress_callback(bar.format_dict)

    os.rename(filename+".tmp", filename)

class SafeUnpickler:
    ignored = []  

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class Unpickler(pickle.Unpickler):
        def is_allowed(self, module, name):
            root = module.split(".",1)[0]
            if root in {"collections", "torch", "_codecs", "numpy"}:
                return True
            if root in {"__builtin__", "__builtins__"}:
                if name in {"list", "tuple", "set", "frozenset", "dict"}:
                    return True
            return False
    
        def find_class(self, module, name):
            if self.is_allowed(module, name):
                return super().find_class(module, name)
            else:
                SafeUnpickler.ignored += [(module, name)]
                return SafeUnpickler.Dummy
    
    def load(*args, **kwargs):
        SafeUnpickler.ignored = []
        return SafeUnpickler.Unpickler(*args, **kwargs).load()

def load_pickle(file, map_location="cpu"):
    try:
        return torch.load(file, map_location=map_location, pickle_module=SafeUnpickler, weights_only=False)
    except:
        raise RuntimeError(f"Failed to unpickle file, {file}\nIgnored types, {str(SafeUnpickler.ignored)}")

def relative_file(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

def block_mapping(labels):
    global MAPPINGS
    name = "MBW_4" if labels == 9 else "MBW_12"

    if name in MAPPINGS:
        return MAPPINGS[name]

    mapping = {}
    labels = []
    with open(relative_file(os.path.join("mappings", f"{name}_mapping.txt"))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            if not dst in labels:
                labels += [dst]
            mapping[src] = labels.index(dst)
            
    MAPPINGS[name] = mapping
    return mapping

def block_mapping_lora(labels):
    global MAPPINGS
    name = "MBW_4_lora" if labels == 9 else "MBW_12_lora"

    if name in MAPPINGS:
        return MAPPINGS[name]

    mapping = block_mapping(labels)

    lora_mapping = {}

    for k, v in mapping.items():
        if k.endswith("bias"):
            continue
        kk = "lora_unet_" + k.rsplit(".",1)[0].replace(".", "_")
        lora_mapping[kk] = v
            
    MAPPINGS[name] = lora_mapping
    return lora_mapping

def lora_mapping(key):
    global MAPPINGS
    name = "SDXL-Base"

    if not name in MAPPINGS:
        mapping = {}
        with open(relative_file(os.path.join("mappings", f"{name}_mapping.txt"))) as file:
            for line in file:
                dst, src = line.strip().split(" TO ")
                if src.startswith("SDXL-Base.UNET.") and src.endswith(".weight"):
                    dst = dst[len("model.diffusion_model."):].rsplit(".", 1)[0].replace(".","_")
                    src = src[len("SDXL-Base.UNET."):].rsplit(".", 1)[0].replace(".","_")
                    mapping[dst] = src
        MAPPINGS[name] = mapping
        
    key, suffix = key.split(".",1)

    for src, dst in MAPPINGS[name].items():
        if key == src or key == "lora_unet_" + src:
            key = key.replace(src, dst)
            break
    
    return key + "." + suffix
    
def get_tile_mask(size, radius):
    width, height = size
    width, height = int(width), int(height)

    radius_x, radius_y = radius
    radius_x, radius_y = int(radius_x), int(radius_y)

    mask = np.ones((height, width), dtype=np.float32)

    radius_x, radius_y = radius

    for i in range(height//2):
        for j in range(width//2):
            weight_x, weight_y = 1, 1
            if i < radius_y:
                weight_y = (i / radius_y)
            if j < radius_x:
                weight_x = (j / radius_x)        
            weight = min(weight_x, weight_y) ** 2
            if weight == 1:
                continue
            mask[i, j] = weight
            mask[i, width-j-1] = weight
            mask[height-i-1, j] = weight
            mask[height-i-1, width-j-1] = weight
    
    return PIL.Image.fromarray(np.uint8(mask*255), mode="L")

def get_tiles(images, tile_size, upscale):
    base_size = int(tile_size)

    tile_positions = []
    tile_images = []
    tile_masks = []

    for img in images:
        img_width, img_height = img.size

        tile_size = int(tile_size / upscale)
        tile_size = int(tile_size - (tile_size % 8))

        if tile_size > min(img_height, img_width):
            tile_size = min(img_height, img_width)

        overlap = tile_size / 4

        count_x = math.ceil(img_width/(tile_size-overlap/2))
        count_y = math.ceil(img_height/(tile_size-overlap/2))
        
        match_width = tile_size == img_width
        match_height = tile_size == img_height

        if match_width and match_height:
            count_x, count_y = 2, 2
            tile_size = int((tile_size + overlap)/2)
        elif match_width:
            count_x = 1
        elif match_height:
            count_y = 1
        else:
            size_x = ((overlap*(count_x-1)) + img_width)/count_x
            size_y = ((overlap*(count_y-1)) + img_height)/count_y
            tile_size = int(max(size_x, size_y))

        interval_x = 0 if count_x == 1 else tile_size - ((count_x*tile_size)-img_width)/(count_x-1)
        interval_y = 0 if count_y == 1 else tile_size - ((count_y*tile_size)-img_height)/(count_y-1)

        positions = []
        tiles = []
        for x in range(count_x):
            for y in range(count_y):
                position_x = int(x*interval_x)
                position_y = int(y*interval_y)
                position = (position_x, position_y, position_x+tile_size, position_y+tile_size)
                positions += [position]
                tiles += [img.crop(position).resize((base_size, base_size))]

        tile_positions += [positions]
        tile_images += [tiles]

        radius_x = (tile_size - interval_x)//2
        radius_y = (tile_size - interval_y)//2

        mask = get_tile_mask((tile_size, tile_size), (radius_x, radius_y))
        tile_masks += [mask]
    
    return tile_images, tile_positions, tile_masks

def assemble_tiles(original_images, tile_images, tile_positions, tile_masks):
    data = zip(original_images, tile_images, tile_positions, tile_masks)
    assembled = []
    for img, tiles, positions, mask in data:
        img_width, img_height = img.size
        img_tensor = TO_TENSOR(img)
        mask_tensor = TO_TENSOR(mask)

        inv_mask = torch.zeros((1, img_height, img_width))
        for (x1, y1, x2, y2) in positions:
            inv_mask[:,y1:y2,x1:x2] += mask_tensor
        inv_mask = 1 - inv_mask.clamp(0,1)

        over_mask = torch.zeros((1, img_height, img_width))
        for (x1, y1, x2, y2) in positions:
            over_mask[:,y1:y2,x1:x2] += mask_tensor + inv_mask[:,y1:y2,x1:x2]
        over_mask = 1 / over_mask.clamp(1,None)

        output = torch.zeros_like(img_tensor)

        for i, ((x1, y1, x2, y2), tile) in enumerate(zip(positions, tiles)):
            w, h = x2-x1, y2-y1
            tile = tile.resize((w,h))
            tile_tensor = TO_TENSOR(tile)
            mask = (mask_tensor + inv_mask[:,y1:y2,x1:x2]) * over_mask[:,y1:y2,x1:x2]
            #masked = torch.cat([tile_tensor, mask])
            #FROM_TENSOR(masked).save(f"{i}.png")
            output[:,y1:y2,x1:x2] += tile_tensor * mask
        
        assembled += [FROM_TENSOR(output)]#.save(f"out.png")

    return assembled