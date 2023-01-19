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

        self.classes = {"UNET": models.UNET, "CLIP": models.CLIP, "VAE": models.VAE, "SR": upscalers.SR}
        
        self.files = {k:{} for k in self.classes}
        self.loaded = {k:{} for k in self.classes}
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

    def get_name(self, file):
        file = file.split(".")[0]
        file = file.split(os.path.sep)[-1]
        return file

    def find_all(self):
        sd_path = os.path.join(self.path, "SD")
        for model in glob.glob(os.path.join(sd_path, "*.st")):
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
        
        sr_path = os.path.join(self.path, "SR")
        for model in glob.glob(os.path.join(sr_path, "*.pth")):
            file = os.path.relpath(model, self.path)
            name = self.get_name(file)
            self.files["SR"][name] = file

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
        
        return self.move(model, name, comp, device)

    def get_unet(self, name, device):
        return self.get_component(name, "UNET", device)
    
    def get_clip(self, name, device):
        return self.get_component(name, "CLIP", device)

    def get_vae(self, name, device):
        return self.get_component(name, "VAE", device)

    def get_upscaler(self, name, device):
        return self.get_component(name, "SR", device)

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
            key = k.removeprefix(f"{model_type}.{comp}.")
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