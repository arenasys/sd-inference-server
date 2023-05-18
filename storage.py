import torch
import os
import glob
import safetensors.torch

import models
import convert
import upscalers
import annotator

MODEL_FOLDERS = {
    "SD": ["SD", "Stable-diffusion", "VAE"],
    "SR": ["SR", "ESRGAN", "RealESRGAN"], 
    "TI": ["TI", "embeddings", os.path.join("..", "embeddings")], 
    "LoRA": ["LoRA"], 
    "HN": ["HN", "hypernetworks"],
    "CN": ["CN"]
}

class ModelStorage():
    def __init__(self, path, dtype, vae_dtype=None):
        self.dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.path = None
        self.set_folder(path)

        self.classes = {"UNET": models.UNET, "CLIP": models.CLIP, "VAE": models.VAE, "SR": upscalers.SR, "LoRA": models.LoRA, "HN": models.HN, "CN": models.ControlNet}
        self.vram_limits = {"UNET": 1, "CLIP": 1, "VAE": 1, "SR": 1}
        self.total_limits = {"UNET": 1, "CLIP": 1, "VAE": 1, "SR": 1}
        self.annotators = {}

        self.files = {k:{} for k in self.classes}
        self.loaded = {k:{} for k in self.classes}
        self.file_cache = {}

        self.embeddings_files = {}
        self.embeddings = {}

        self.find_all()

    def set_folder(self, path):
        if path != self.path:
            self.clear_file_cache()
        self.path = path
            
    def get_models(self, folder, ext, recursive=True):
        files = []
        for f in MODEL_FOLDERS[folder]:
            path = os.path.abspath(os.path.join(self.path, f))
            if recursive:
                for e in ext:
                    files += glob.glob(os.path.join(path, "**" + os.sep + e), recursive=True)
            else:
                for e in ext:
                    files += glob.glob(os.path.join(path, e))
        return [f for f in files if not os.path.sep + "_" in f]

    def clear_file_cache(self):
        self.file_cache = {}

    def reset(self):
        self.embeddings = {}
        for c in self.loaded:
            for m in list(self.loaded[c].keys()):
                self.loaded[c][m].to("cpu")
                del self.loaded[c][m]
        torch.cuda.empty_cache()
        self.file_cache = {}
        self.find_all()

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

    def load(self, model, device):
        model.to(device)

    def unload(self, model):
        model.to("cpu")
        torch.cuda.empty_cache()

    def load(self, model, device):
        model.to(device)

    def unload(self, model):
        model.to("cpu")
        torch.cuda.empty_cache()

    def move(self, model, name, comp, device):
        dtype = self.dtype
        if comp in {"VAE"}:
            dtype = self.vae_dtype

        if model.device == device:
            return model.to(dtype)

        if comp in self.total_limits:
            if str(device) == "cpu":
                self.enforce_total_limit(name, comp, self.total_limits[comp]-1)
            else:
                self.enforce_vram_limit(name, comp, self.vram_limits[comp]-1)
                self.enforce_total_limit(name, comp, self.total_limits[comp]-1)
        
        return model.to(device, dtype)

    def get_name(self, file):
        return os.path.relpath(file, self.path)

    def find_all(self):
        self.files = {k:{} for k in self.classes}

        standalone = {k:{} for k in ["UNET", "CLIP", "VAE"]}
        for file in self.get_models("SD", ["*.st", "*.safetensors", "*.ckpt", "*.pt"]):
            if ".unet." in file:
                name = self.get_name(file)
                standalone["UNET"][name] = file
            elif ".clip." in file:
                name = self.get_name(file)
                standalone["CLIP"][name] = file
            elif ".vae." in file:
                name = self.get_name(file)
                standalone["VAE"][name] = file
            else:
                name = self.get_name(file)
                self.files["UNET"][name] = file
                self.files["CLIP"][name] = file
                self.files["VAE"][name] = file
        for comp in standalone:
            for name, file in standalone[comp].items():
                self.files[comp][name] = file
        
        for file in self.get_models("SR", ["*.pth"]):
            name = self.get_name(file)
            self.files["SR"][name] = file

        for file in self.get_models("TI", ["*.pt", "*.safetensors", "*.bin"]):
            name = self.get_name(file)
            activation = name.rsplit(".", 1)[0].rsplit(os.path.sep,1)[-1]
            if activation in self.embeddings:
                continue

            self.embeddings_files[name] = file

            ti = self.load_file(file, "TI")["TI"]

            if "string_to_param" in ti:
                vectors = ti["string_to_param"]["*"]
            elif 'emb_params' in ti:
                vectors = ti['emb_params']
            elif len(ti) == 1:
                vectors = ti[list(ti.keys())[0]]
            else:
                raise Exception("Unknown TI format")
            
            vectors.requires_grad = False
            self.embeddings[activation] = vectors

        for file in self.get_models("LoRA", ["*.safetensors", "*.pt"]):
            name = self.get_name(file)
            self.files["LoRA"][name] = file

        for file in self.get_models("HN", ["*.pt"]):
            name = self.get_name(file)
            self.files["HN"][name] = file

        for file in self.get_models("CN", ["*.safetensors", "*.pth"], False):
            name = self.get_name(file)
            self.files["CN"][name] = file

    def get_component(self, name, comp, device):
        if name in self.loaded[comp]:
            return self.move(self.loaded[comp][name], name, comp, device)
        
        if not name in self.files[comp]:
            raise ValueError(f"unknown {comp}: {name}")
        
        file = self.files[comp][name]
        
        if not file in self.file_cache:
            self.file_cache[file] = self.load_file(file, comp)

        if comp in self.file_cache[file]:
            dtype = self.vae_dtype if comp == "VAE" else self.dtype
            model = self.classes[comp].from_model(name, self.file_cache[file][comp], dtype)
        else:
            raise ValueError(f"model doesnt contain a {comp}: {name}")

        self.loaded[comp][name] = model
        return self.move(model, name, comp, device)

    def get_state_dict(self, file, comp):
        if not file in self.file_cache:
            self.file_cache[file] = self.load_file(file, comp)
        return self.file_cache[file][comp]
    
    def get_filename(self, name, comp):
        if not name in self.files[comp]:
            raise ValueError(f"unknown {comp}: {name}")
        return self.files[comp][name]

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
        for lora in self.files["LoRA"]:
            if os.path.sep + name + "." in lora:
                name = lora
                break
        return self.get_component(name, "LoRA", device)

    def get_hypernetwork(self, name, device):
        for hn in self.files["HN"]:
            if os.path.sep + name + "." in hn:
                name = hn
                break
        return self.get_component(name, "HN", device)

    def get_controlnet(self, name, device):
        for cn in self.files["CN"]:
            if os.path.sep + name + "." in cn:
                name = cn
                break
        return self.get_component(name, "CN", device)
    
    def get_controlnet_annotator(self, name, device, dtype):
        if not name in self.annotators:
            self.annotators[name] = annotator.annotators[name](os.path.join(self.path, "CN", "annotators"))
        return self.annotators[name].to(device, dtype)
    
    def load_file(self, file, comp):
        print(f"LOADING {file.rsplit(os.path.sep, 1)[-1]}...")

        if comp in ["UNET", "CLIP", "VAE"] and file.rsplit(".",1)[-1] in {"safetensors", "ckpt", "pt"}:
            state_dict = convert.convert_checkpoint(file)
            return self.parse_model(state_dict)
        
        out = {}
        if file.endswith(".st") or file.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(file)
            if "metadata.model_type" in state_dict:
                out = self.parse_model(state_dict)
            else:
                out = {comp: state_dict}
        else:
            out = {comp: torch.load(file, map_location="cpu")}

        if comp in out and comp == "CN":
            out[comp] = convert.CN_convert(out[comp])
            
        return out


    def parse_model(self, state_dict):
        metadata = {}
        for k in state_dict:
            if k.startswith("metadata."):
                kk = k.split(".", 1)[1]
                metadata[kk] = ''.join([chr(c) for c in state_dict[k]])

        model_type = metadata["model_type"].split("-",1)[0]

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
            metadata["dtype"] = dtype
            sub_state_dicts[m]['metadata'] = metadata.copy()
        
        return sub_state_dicts