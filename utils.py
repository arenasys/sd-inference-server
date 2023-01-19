import PIL
import torch
import torchvision.transforms as transforms

TO_TENSOR = transforms.ToTensor()
FROM_TENSOR = transforms.ToPILImage()

def preprocess_images(images):
    def process(image):
        w, h = image.size
        w, h = w - w % 32, h - h % 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = TO_TENSOR(image).to(torch.float32)
        return 2.0 * image[None, :] - 1.0
    return torch.cat([process(i) for i in images])

def postprocess_images(images):
    def process(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        return FROM_TENSOR(image)
    return [process(i) for i in images]

def preprocess_masks(masks):
    def process(mask):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = w - w % 32, h - h % 32
        mask = mask.resize((w // 8, h // 8), resample=PIL.Image.LANCZOS)
        mask = 1 - TO_TENSOR(mask).to(torch.float32)
        return torch.cat([mask]*4)[None, :]
    return torch.cat([process(m) for m in masks])

def encode_images(vae, seeds, images):
    with torch.autocast(vae.autocast(), vae.dtype):
        images = preprocess_images(images).to(vae.device)
        noise = singular_noise(seeds, images.shape[3] // 8, images.shape[2] // 8, vae.device)

        dists = vae.encode(images).latent_dist
        mean = torch.stack([dists.mean[i%len(images)] for i in range(len(seeds))])
        std = torch.stack([dists.std[i%len(images)] for i in range(len(seeds))])

        latents = mean + std * noise
        
        return latents

def decode_images(vae, latents):
    with torch.autocast(vae.autocast(), vae.dtype):
        latents = latents.clone().detach().to(vae.device).to(vae.dtype)
        images = vae.decode(latents).sample
        return postprocess_images(images)

def get_latents(vae, seeds, images):
    if type(images) == torch.Tensor:
        return images.to(vae.device)
    elif type(images) == list:
        return encode_images(vae, seeds, images)

def get_masks(device, masks):
    if type(masks) == torch.Tensor:
        return masks      
    elif type(masks) == list:
        return preprocess_masks(masks).to(device)

def apply_inpainting(images, originals, masks, extents):
    outputs = [None] * len(images)
    for i in range(len(images)):
        image = images[i]
        original = originals[i%len(originals)]
        mask = masks[i%len(masks)]
        extent = extents[i%len(extents)]
        
        ew, eh = extent[2]-extent[0], extent[3]-extent[1]
        if (ew, eh) != images[i].size:
            image = image.resize((ew, eh))
            mask = mask.resize((ew, eh))
        
        outputs[i] = original.copy()
        outputs[i].paste(image, extent, mask)

        #draw = PIL.ImageDraw.Draw(outputs[i])
        #draw.rectangle(extent, width=2, outline=(255,0,0))

    return outputs

def prepare_inpainting(originals, masks, padding, mask_blur):
    def pad_extent(extent, p, w, h):
        x1, y1, x2, y2 = extent

        # add padding
        x1, y1 = max(x1-p, 0), max(y1-p, 0)
        x2, y2 = min(x2+p, w), min(y2+p, h)

        # scale to image aspect ratio
        ew = (x2-x1)
        eh = (y2-y1)

        if ew > eh:
            while y2-y1 != ew:
                o = ew-(y2-y1)
                y1 = max(y1-(o//2), 0)
                y2 = min(y2+(o-o//2), h)
        else:
            while x2-x1 != eh:
                o = (eh-(x2-x1))//2
                x1 = max(x1-(o//2), 0)
                x2 = min(x2+(o-o//2), w)

        return x1, y1, x2, y2

    w, h = originals[0].size[0], originals[0].size[1]
    if padding != None:
        extents = [pad_extent(mask.getbbox(), padding, w, h) for mask in masks]
    else:
        # dont resize for inpainting (extent is the entire image)
        extents = [(0, 0, w, h) for _ in masks]

    # crop + resize masks according to their extent
    masks = [m for m in masks]
    for i in range(len(masks)):
        extent = extents[i]
        if extent != (0,0,w,h):
            masks[i] = masks[i].crop(extent)
            mw, mh = masks[i].size
            scale = min(w/mw, h/mh)
            mw, mh = int(mw*scale), int(mh*scale)
            masks[i] = masks[i].resize((mw, mh), )
            masks[i] = masks[i].filter(PIL.ImageFilter.GaussianBlur(mask_blur))

    # crop + resize images according to their extent
    images = [i for i in originals]
    for i in range(len(images)):
        extent = extents[i%len(extents)]
        if extent != (0,0,w,h):
            images[i] = images[i].crop(extent)
            iw, ih = images[i].size
            scale = min(w/iw, h/ih)
            iw, ih = int(iw*scale), int(ih*scale)
            images[i] = images[i].resize((iw, ih))

    return images, masks, extents

def cast_state_dict(state_dict, dtype):
    for k in state_dict:
        if type(k) == torch.Tensor and k.dtype in {torch.float16, torch.float32}:
            state_dict[k] = state_dict[k].to(dtype)
    return state_dict
    
class NoiseSchedule():
    def __init__(self, seeds, width, height, device):
        self.seeds = seeds
        self.shape = (4, int(height), int(width))
        self.device = device
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
        generator = torch.Generator(self.device)
        noises = []
        for i in range(len(self.seeds)):
            generator.manual_seed(self.seeds[i])
            noises += [[]]

            for _ in range(self.steps+1):
                noise = torch.randn(self.shape, generator=generator, device=self.device)
                noises[i] += [noise]

        combined = [torch.stack(n) for n in zip(*noises)]
        self.noise = combined
    
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
    shape = (4, int(height), int(width))
    generator = torch.Generator(device)
    noises = []
    for i in range(len(seeds)):
        generator.manual_seed(seeds[i])
        noises += [[]]
        noises[i] = torch.randn(shape, generator=generator, device=device)
        
    combined = torch.stack(noises)
    return combined

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