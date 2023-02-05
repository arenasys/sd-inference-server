import torch
import os
import glob
import safetensors.torch

import models
import upscalers

class ModelStorage():
    def __init__(self, path, dtype, vae_dtype=None):
        self.path = path
        self.dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.classes = {"UNET": models.UNET, "CLIP": models.CLIP, "VAE": models.VAE, "SR": upscalers.SR, "LoRA": models.LoRA, "HN": models.HN}
        self.vram_limits = {"UNET": 1, "CLIP": 1, "VAE": 1, "SR": 1}
        self.total_limits = {"UNET": 1, "CLIP": 1, "VAE": 1, "SR": 1}

        self.files = {k:{} for k in self.classes}
        self.loaded = {k:{} for k in self.classes}
        self.file_cache = {}

        self.embeddings = {}

        self.find_all()

    def clear_file_cache(self):
        self.file_cache = {}

    def enforce_vram_limit(self, allowed, comp, limit):
        found = 0
        for m in self.loaded[comp]:
            if m == allowed:
                continue
            if str(self.loaded[comp][m].device) != "cpu":
                if found >= limit:
                    self.loaded[comp][m].to("cpu")
                else:
                    found += 1
        torch.cuda.empty_cache()

    def enforce_total_limit(self, allowed, comp, limit):
        found = 0
        for m in list(self.loaded[comp].keys()):
            if m == allowed:
                continue
            if found >= limit:
                self.loaded[comp][m].to("cpu")
                del self.loaded[comp][m]
            else:
                found += 1
        torch.cuda.empty_cache()

    def enforce_network_limit(self, used, comp):
        # networks cant have a hard limit since you can use an arbitrary number of them
        # so they are "decayed" from gpu -> cpu -> disk as they are left unused
        for m in list(self.loaded[comp].keys()):
            if m in used:
                continue
            if str(self.loaded[comp][m].device) != "cpu":
                self.loaded[comp][m].to("cpu")
            else:
                del self.loaded[comp][m]

    def move(self, model, name, comp, device):
        if model.device == device:
            return model

        if comp in self.total_limits:
            if str(device) == "cpu":
                self.enforce_total_limit(name, comp, self.total_limits[comp]-1)
            else:
                self.enforce_vram_limit(name, comp, self.vram_limits[comp]-1)
                self.enforce_total_limit(name, comp, self.total_limits[comp]-1)
        
        return model.to(device)

    def get_name(self, file):
        file = file.split(".")[0]
        file = file.split(os.path.sep)[-1]
        return file

    def find_all(self):
        for model in glob.glob(os.path.join(self.path, "SD", "*.st")):
            file = os.path.relpath(model, self.path)

            if ".unet." in file:
                name = self.get_name(file)
                self.files["UNET"][name] = file
            elif ".clip." in file:
                name = self.get_name(file)
                self.files["CLIP"][name] = file
            elif ".vae." in file:
                name = self.get_name(file)
                self.files["VAE"][name] = file
            else:
                name = self.get_name(file)
                self.files["UNET"][name] = file
                self.files["CLIP"][name] = file
                self.files["VAE"][name] = file
        
        for model in glob.glob(os.path.join(self.path, "SR", "*.pth")):
            file = os.path.relpath(model, self.path)
            name = self.get_name(file)
            self.files["SR"][name] = file

        self.embeddings = {}
        for model in glob.glob(os.path.join(self.path, "TI", "*.pt")):
            file = os.path.relpath(model, self.path)
            name = self.get_name(file)
            vectors = torch.load(model)["string_to_param"]["*"]
            vectors.requires_grad = False
            self.embeddings[name] = vectors

        for model in glob.glob(os.path.join(self.path, "LoRA", "*.safetensors")):
            file = os.path.relpath(model, self.path)
            name = self.get_name(file)
            self.files["LoRA"][name] = file

        for model in glob.glob(os.path.join(self.path, "HN", "*.pt")):
            file = os.path.relpath(model, self.path)
            name = self.get_name(file)
            self.files["HN"][name] = file

    def get_component(self, name, comp, device):
        if name in self.loaded[comp]:
            return self.move(self.loaded[comp][name], name, comp, device)
        
        if not name in self.files[comp]:
            raise ValueError(f"ERROR unknown {comp}: {name}")
        
        file = self.files[comp][name]
        
        if not file in self.file_cache:
            self.file_cache[file] = self.load_file(file, comp)

        if comp in self.file_cache[file]:
            dtype = self.vae_dtype if comp == "VAE" else self.dtype
            model = self.classes[comp].from_model(self.file_cache[file][comp], dtype)
        else:
            raise ValueError(f"ERROR model doesnt contain a {comp}: {model}")

        self.loaded[comp][name] = model
        return self.move(model, name, comp, device)

    def get_unet(self, name, device):
        unet = self.get_component(name, "UNET", device)
        return unet

    def get_clip(self, name, device):
        clip = self.get_component(name, "CLIP", device)
        clip.textual_inversions = {}
        return clip

    def get_vae(self, name, device):
        return self.get_component(name, "VAE", device)

    def get_upscaler(self, name, device):
        return self.get_component(name, "SR", device)

    def get_embeddings(self, device):
        for k in self.embeddings:
            self.embeddings[k] = self.embeddings[k].to(device)
        return self.embeddings

    def get_lora(self, name, device):
        return self.get_component(name, "LoRA", device)

    def get_hypernetwork(self, name, device):
        return self.get_component(name, "HN", device)

    def load_file(self, file, comp):
        print(f"LOADING {file}...")
        file = os.path.join(self.path, file)

        if file.endswith(".st") or file.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(file)
            if "metadata.model_type" in state_dict:
                return self.parse_model(state_dict)
            else:
                return {comp: state_dict}
        else:
            return {comp: torch.load(file)}

    def parse_model(self, state_dict):
        model_type = ''.join([chr(c) for c in state_dict["metadata.model_type"]])

        sub_state_dicts = {}
        for k in list(state_dict.keys()):
            if k.startswith("metadata."):
                continue
            comp = k.split(".")[1]
            key = k[len(f"{model_type}.{comp}."):]
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