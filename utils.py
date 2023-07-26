import PIL
import torch
import torchvision.transforms as transforms
import requests
import tqdm
import os
import time
import pickle

DIRECTML_AVAILABLE = False
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except:
    pass

TO_TENSOR = transforms.ToTensor()
FROM_TENSOR = transforms.ToPILImage()

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
        masked[i].putalpha(mask)

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
    
    if not padding:
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

def cast_state_dict(state_dict, dtype):
    for k in state_dict:
        if type(state_dict[k]) == torch.Tensor and state_dict[k].dtype != dtype and state_dict[k].dtype in {torch.float16, torch.float32, torch.bfloat16}:
            tmp = state_dict[k].clone().detach()
            state_dict[k] = tmp.to('cpu', dtype=dtype)
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
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()

        torch.cuda.synchronize()
        print(self.start.elapsed_time(self.end))

def download(url, filename, callback):
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    desc = filename.rsplit(os.path.sep)[-1]
    resp = requests.get(url, stream=True, timeout=10)
    total = int(resp.headers.get('content-length', 0))
    last = None
    with open(filename+".tmp", 'wb') as file, tqdm.tqdm(desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            if not last or time.time() - last > 0.5:
                last = time.time()
                callback(bar.format_dict)

    os.rename(filename+".tmp", filename)

class SafeUnpickler:
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            root = module.split(".",1)[0]
            if root in {"collections", "torch"}:
                return super().find_class(module, name)
            else:
                print("IGNORE", module, name)
                return SafeUnpickler.Dummy

def load_pickle(file, map_location="cpu"):
    return torch.load(file, map_location=map_location, pickle_module=SafeUnpickler)