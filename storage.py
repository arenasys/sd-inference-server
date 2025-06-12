import torch
import os
import glob
import safetensors.torch
import gc
import json

import models
import convert
import upscalers
import annotator
import controlnet
import segmentation
import utils

MODEL_FOLDERS = {
    "SD": ["SD", "Stable-diffusion", "VAE"],
    "SR": ["SR", "ESRGAN", "RealESRGAN"], 
    "TI": ["TI", "embeddings", os.path.join("..", "embeddings")], 
    "LoRA": ["LoRA", "LyCORIS"],
    "CN": ["CN"],
    "Detailer": ["Detailer"],
}

MODEL_TYPE_NAMES = {
    "unknown": "Unknown",
    "lora": "LoRA",
    "locon": "LoCon",
    "loha": "LoHa",
    "lokr": "LoKr",
    "ia3": "IA3",
    "full": "Full"
}

MODEL_TYPE_IDENTIFIERS = {
    "lora": ["lora_up"],
    "loha": ["hada_w1_a"],
    "lokr": ["lokr_w1", "lokr_w1_a"],
    "full": ["diff"],
    "ia3": ["on_input"]
}

class ModelStorage():
    def __init__(self, path, dtype, vae_dtype=None, cache=0):
        self.dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.path = None
        self.set_folder(path)

        self.classes = {"UNET": models.UNET, "CLIP": models.CLIP, "VAE": models.VAE, "SR": upscalers.SR, "LoRA": models.LoRA, "CN": models.ControlNet, "AN": torch.nn.Module, "Detailer": models.Detailer}
        self.vram_limits = {"UNET": 0, "CLIP": 0, "VAE": 0, "SR": 0}
        self.ram_limits = {"UNET": cache, "CLIP": cache, "VAE": cache, "SR": cache, "LoRA": cache, "CN": cache, "AN": cache, "Detailer": cache}

        self.files = {k:{} for k in self.classes}
        self.loaded = {k:{} for k in self.classes}
        self.order = {k:[] for k in self.classes}
        self.file_cache = {}

        self.uncap_ram = False

        self.embeddings_files = {}
        self.embeddings = {}

        self.model_types = {}

        self.find_all()

    def set_folder(self, path):
        if path != self.path:
            self.clear_file_cache()
        self.path = path

    def get_folder(self, type):
        folders = [os.path.join(self.path, f) for f in MODEL_FOLDERS[type]]
        folders = [f for f in folders if os.path.exists(f)]
        return folders[0]

    def get_models(self, folder, ext, recursive=True):
        files = []
        diffusers = []
        for f in MODEL_FOLDERS[folder]:
            tmp = []
            path = os.path.abspath(os.path.join(self.path, f))
            if recursive:
                for e in ext:
                    tmp += glob.glob(os.path.join(path, "**" + os.sep + e), recursive=True)
                if folder == "SD":
                    folders = glob.glob(os.path.join(path, "**" + os.sep + "model_index.json"), recursive=True)
                    diffusers += [f.rsplit(os.path.sep, 1)[0] for f in folders]
            else:
                for e in ext:
                    tmp += glob.glob(os.path.join(path, e))
            
            for file in tmp:
                rel = os.path.relpath(file, path)
                if rel.startswith("_") or os.path.sep + "_" in rel:
                   continue
                files += [file]
        
        files = [f for f in files if not any([d + os.path.sep in f for d in diffusers])] + diffusers
        return files

    def do_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def clear_file_cache(self):
        self.file_cache = {}
        self.do_gc()

    def clear_vram(self):
        for c in self.loaded:
            for m in list(self.loaded[c].keys()):
                if not m in self.loaded[c]:
                    continue
                if str(self.loaded[c][m].device) != "cpu":
                    if c in self.ram_limits:
                        self.enforce_limit(c, m, "cpu")
                    #print("CLEAR", c, m, "TO RAM")
                    self.unload(self.loaded[c][m])
        self.do_gc()

    def reset(self):
        self.embeddings = {}
        for c in self.loaded:
            for m in list(self.loaded[c].keys()):
                del self.loaded[c][m]
        self.file_cache = {}
        self.do_gc()
        self.find_all()

    def set_ram_limit(self, count):
        for comp in {"UNET", "CLIP", "VAE"}:
            self.ram_limits[comp] = count

    def enforce_limit(self, comp, current, device):
        allowed_vram = self.vram_limits.get(comp, 0)
        allowed_ram = self.ram_limits.get(comp, 0)

        found_vram = 0
        found_ram = 0

        if str(device) == "cpu":
            found_ram += 1
        else:
            found_vram += 1
        
        for m in self.order[comp]:
            if not m in self.loaded[comp]:
                continue
            
            in_ram = str(self.loaded[comp][m].device) == "cpu"
            if m == current:
                continue

            if found_vram >= allowed_vram and not in_ram:
                if found_ram < allowed_ram:
                    #print("ENFORCE", comp, m, "TO RAM")
                    self.loaded[comp][m] = self.loaded[comp][m].to("cpu")
                    found_ram += 1
                else:
                    #print("ENFORCE", comp, m, "TO DISK")
                    del self.loaded[comp][m]
                self.do_gc()
                continue

            if found_ram >= allowed_ram and in_ram and not self.uncap_ram:
                #print("ENFORCE", comp, m, "TO DISK")
                del self.loaded[comp][m]
                self.do_gc()
                continue
            
            if in_ram:
                found_ram += 1
            else:
                found_vram += 1

    def enforce_network_limit(self, used, comp):
        # networks cant have a hard limit since you can use an arbitrary number of them
        # so they are "decayed" from gpu -> cpu -> disk as they are left unused
        for m in list(self.loaded[comp].keys()):
            if str(self.loaded[comp][m].device) == "cpu":
                continue
            if any([os.path.sep+u+"." in m or m == u for u in used]):
                continue
            #print("NET ENFORCE", m, "TO DISK")
            del self.loaded[comp][m]
            self.do_gc()

    def enforce_controlnet_limit(self, used):
        for m in list(self.loaded["CN"].keys()):
            if str(self.loaded["CN"][m].device) == "cpu":
                continue
            if not m in used:
                #print("CN ENFORCE", m, "TO DISK")
                del self.loaded["CN"][m]
                self.do_gc()

    def clear_modified(self):
        # static network mode will merge models into the UNET/CLIP
        # so reload from disk
        for comp in {"UNET", "CLIP"}:
            for model in self.loaded[comp]:
                self.loaded[comp][model] = None
            self.loaded[comp] = {}
        self.clear_file_cache()

    def clear_annotators(self, allowed=[]):
        for m in list(self.loaded["AN"].keys()):
            if str(self.loaded["AN"][m].device) == "cpu":
                continue
            if not m in allowed:
                #print("AN ENFORCE", m, "TO DISK")
                del self.loaded["AN"][m]
                self.do_gc()

    def load(self, model, device, dtype=None):
        if dtype:
            model.to(device, dtype)
        else:
            model.to(device)
        self.do_gc()

    def unload(self, model):
        model.to("cpu")
        self.do_gc()

    def add(self, comp, name, model):
        self.loaded[comp][name] = model
        if name in self.order[comp]:
            self.order[comp].remove(name)
        self.order[comp].insert(0, name)

    def remove(self, comp, name):
        if name in self.loaded[comp]:
            if hasattr(self.loaded[comp][name], "additional"):
                self.loaded[comp][name].additional.reset()
            del self.loaded[comp][name]
            
        self.do_gc()

    def reset_merge(self, comps):
        self.uncap_ram = False
        for comp in comps:
            for name in list(self.loaded[comp].keys()):
                if not "." in name:
                    self.remove(comp, name)

    def move(self, model, name, comp, device):
        dtype = self.dtype
        if comp in {"VAE"}:
            dtype = self.vae_dtype

        if comp in self.vram_limits:
            self.enforce_limit(comp, name, device)

        model = model.to(device, dtype)

        return model

    def get_name(self, file):
        return os.path.relpath(file, self.path)
    
    def get_lycoris_type(self, file):
        algo = "unknown"
        try:
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                module = metadata.get("ss_network_module", "")
                args = {}
                try:
                    args = json.loads(metadata.get("ss_network_args", "{}"))
                except:
                    pass

                if "algo" in args:
                    algo = args["algo"]
                if "conv_dim" in args and (algo == "lora" or "networks.lora" in module):
                    algo = "locon"
                if algo == "unknown" and "networks.lora" in module:
                    algo = "lora"

                if algo == "unknown" or algo == "lora":
                    suffices = set([f.split(".", 1)[-1].split(".", 1)[0] for f in f.keys()])

                    for a, ids in MODEL_TYPE_IDENTIFIERS.items():
                        if any([i in suffices for i in ids]):
                            algo = a
                            break

                    if algo == "lora":
                        for k in f.keys():
                            if "conv" in k:
                                algo = "locon"
                                break
        except:
            pass
    
        return MODEL_TYPE_NAMES.get(algo, None)
    
    def get_metadata(self, name):
        file = None
        for comp in self.files:
            if name in self.files[comp]:
                file = self.files[comp][name]
                break
        
        if not file:
            raise ValueError(f"unknown model: {name}")

        metadata = {}
        try:
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
        except:
            pass

        return metadata

    def find_all(self):
        self.files = {k:{} for k in self.classes}

        standalone = {k:{} for k in ["UNET", "CLIP", "VAE"]}
        for file in self.get_models("SD", ["*.safetensors", "*.ckpt", "*.pt"]):
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
        
        for file in self.get_models("SR", ["*.pth", "*.pt", "*.safetensors"]):
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
            elif 'clip_g' in ti and 'clip_l' in ti:
                vectors = torch.cat([ti["clip_g"], ti["clip_l"]], dim=1)
            else:
                raise Exception(f"Unknown TI format in {name}")
            
            vectors.requires_grad = False
            self.embeddings[activation] = vectors

        for file in self.get_models("LoRA", ["*.safetensors", "*.pt"]):
            name = self.get_name(file)
            self.files["LoRA"][name] = file
            if not name in self.model_types:
                self.model_types[name] = self.get_lycoris_type(file)

        for file in self.get_models("CN", ["*.safetensors", "*.pth"], False):
            name = self.get_name(file)
            self.files["CN"][name] = file

        for file in self.get_models("Detailer", ["*.safetensors", "*.pt"]):
            name = self.get_name(file)
            self.files["Detailer"][name] = file

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
        
        self.add(comp, name, model)
        return self.move(model, name, comp, device)

    def get_state_dict(self, file, comp):
        if not file in self.file_cache:
            self.file_cache[file] = self.load_file(file, comp)
        return self.file_cache[file][comp]
    
    def get_filename(self, name, comp):
        if not name in self.files[comp]:
            raise ValueError(f"unknown {comp}: {name}")
        return self.files[comp][name]

    def check_attached_networks(self, name, comp, allowed):
        if name in self.loaded[comp]:
            if self.loaded[comp][name].additional.need_reset(allowed):
                self.remove(comp, name)
                self.do_gc() 

    def get_unet(self, name, device, nets={}):
        self.check_attached_networks(name, "UNET", nets)
        unet = self.get_component(name, "UNET", device)
        if str(device) != "cpu":
            unet.determine_type()
        return unet

    def get_clip(self, name, device, nets={}):
        self.check_attached_networks(name, "CLIP", nets)
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

    def get_controlnet(self, name, device, callback):
        file, _ = controlnet.get_controlnet(name, os.path.join(self.path, "CN"), callback)
        file = os.path.join("CN", file)

        if name in self.loaded["CN"]:
            return self.move(self.loaded["CN"][name], name, "CN", device)

        state_dict = self.load_file(os.path.join(self.path, file), "CN")
        if not "CN" in state_dict:
            raise ValueError(f"model doesnt contain a CN: {file}")
        
        model = models.ControlNet.from_model(name, state_dict["CN"], self.dtype)
        self.add("CN", name, model)

        return self.move(model, name, "CN", device)
    
    def get_controlnet_annotator(self, name, device, dtype, callback):
        if not name in annotator.annotators:
            return None
        
        def download(url, file):
            utils.download(url, file, callback)

        if not name in self.loaded["AN"]:
            print(f"LOADING {name} ANNOTATOR...")
            model = annotator.annotators[name](os.path.join(self.path, "CN", "annotators"), download).to(device, dtype)
            self.add("AN", name, model)
        else:
            model = self.loaded["AN"][name]
        
        return self.move(model, name, "AN", device)
    
    def get_segmentation_annotator(self, name, device, callback):
        if not name in self.loaded["AN"]:
            print(f"LOADING {name} ANNOTATOR...")
            folder = os.path.join(self.path, "SEGMENTATION")
            model = segmentation.get_predictor(name, folder, callback)
            self.add("AN", name, model)
        else:
            model = self.loaded["AN"][name]
        return self.move(model, name, "AN", device)
    
    def get_detailer(self, name, device):
        return self.get_component(name, "Detailer", device)

    def load_file(self, file, comp):
        if not comp == "TI":
            print(f"LOADING {file.rsplit(os.path.sep, 1)[-1]}...")

        if comp in ["UNET", "CLIP", "VAE", "Checkpoint"]:
            state_dict, metadata = convert.convert(file)
            return self.parse_model(state_dict, metadata)
        
        if comp == "Detailer":
            return {comp: models.Detailer(file)}
        
        if file.endswith(".safetensors"):
            out = {comp: safetensors.torch.load_file(file)}
        else:
            out = {comp: utils.load_pickle(file)}

        if comp in out and comp == "CN":
            out[comp] = convert.CN_convert(out[comp])
            
        return out

    def parse_model(self, state_dict, metadata):
        model_type = metadata["model_type"]

        sub_state_dicts = {}
        for k in list(state_dict.keys()):
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