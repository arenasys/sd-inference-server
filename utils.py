import PIL
import torch
import torchvision.transforms as transforms
import os
import glob
from safetensors.torch import load_file

import models

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

def noise(seeds, width, height, device):
    shape = (4, height, width)
    generator = torch.Generator(device)
    noises = []
    for i in range(len(seeds)):
        generator.manual_seed(seeds[i])
        noises += [[]]
        noises[i] = torch.randn(shape, generator=generator, device=device)
        
    combined = torch.stack(noises)
    return combined


class ModelStorage():
    def __init__(self, path, dtype, vae_dtype=None):
        self.path = path
        self.dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.files = {"UNET": {}, "CLIP": {}, "VAE": {}}
        self.loaded = {"UNET": {}, "CLIP": {}, "VAE": {}}
        self.classes = {"UNET": models.UNET, "CLIP": models.CLIP, "VAE": models.VAE}
        self.file_cache = {}

        self.find_all()

    def clear_cache(self):
        self.file_cache = {}

    def move(self, model, name, comp, device):
        if model.device == device:
            return model

        # TODO: expand on this to allow cacheing
        
        self.loaded[comp] = {name: model}
        torch.cuda.empty_cache()

        return model.to(device)

    def find_all(self):
        for model in glob.glob(os.path.join(self.path, "*.st")):
            model = os.path.relpath(model, self.path)

            if ".unet." in model:
                name = model.split(".unet.")[0]
                self.files["UNET"][name] = model
            elif ".clip." in model:
                name = model.split(".clip.")[0]
                self.files["CLIP"][name] = model
            elif ".vae." in model:
                name = model.split(".vae.")[0]
                self.files["VAE"][name] = model
            else:
                name = model.removesuffix(".st")
                self.files["UNET"][name] = model
                self.files["CLIP"][name] = model
                self.files["VAE"][name] = model

    def get_component(self, name, comp, device):
        if name in self.loaded[comp]:
            return self.move(self.loaded[comp][name], name, comp, device)
        
        if not name in self.files[comp]:
            raise ValueError(f"ERROR unknown {comp}: {name}")
        
        file = self.files[comp][name]
        
        if not file in self.file_cache:
            self.file_cache[file] = self.load_model(file)

        if comp in self.file_cache[file]:
            dtype = self.vae_dtype if comp == "VAE" else self.dtype
            model = self.classes[comp].from_model(self.file_cache[file][comp], dtype)
        else:
            raise ValueError(f"ERROR model doesnt contain a {comp}: {model}")
        
        return self.move(model, name, comp, device)

    def get_unet(self, name, device):
        return self.get_component(name, "UNET", device)
    
    def get_clip(self, name, device):
        return self.get_component(name, "CLIP", device)

    def get_vae(self, name, device):
        return self.get_component(name, "VAE", device)

    def load_model(self, model):
        print(f"LOADING {model}...")

        state_dict = load_file(os.path.join(self.path, model))

        model_type = list(state_dict.keys())[0].split(".")[0]

        sub_state_dicts = {}
        for k in list(state_dict.keys()):
            comp = k.split(".")[1]
            key = k[len(model_type)+len(comp)+2:]
            if not comp in sub_state_dicts:
                sub_state_dicts[comp] = {}
            sub_state_dicts[comp][key] = state_dict[k]
            del state_dict[k]

        for m in sub_state_dicts:
            dtype = None
            for k in sub_state_dicts[m]:
                t = sub_state_dicts[m][k]
                if type(t) == torch.Tensor and t.dtype in [torch.float16, torch.float32]:
                    dtype = t.dtype
                    break
            sub_state_dicts[m]['metadata'] = dict(model_type=model_type, dtype=dtype)
        
        return sub_state_dicts

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