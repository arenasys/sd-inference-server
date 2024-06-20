
import torch
import base64
import os
import tqdm
import safetensors
import contextlib
import gc

import models
import utils
import storage
import lora
import convert

block_4_labels = ["DOWN0","DOWN1","DOWN2","DOWN3","MID","UP0","UP1","UP2","UP3"]
block_4_keys = {
    "DOWN0": ["time_embed.", "input_blocks.0.", "input_blocks.1.", "input_blocks.2.", "input_blocks.3."],
    "DOWN1": ["input_blocks.4.", "input_blocks.5.", "input_blocks.6."],
    "DOWN2": ["input_blocks.7.", "input_blocks.8.", "input_blocks.9."],
    "DOWN3": ["input_blocks.10.", "input_blocks.11."],
    "MID": ["middle_block."],
    "UP0": ["output_blocks.0.", "output_blocks.1.", "output_blocks.2."],
    "UP1": ["output_blocks.3.", "output_blocks.4.", "output_blocks.5."],
    "UP2": ["output_blocks.6.", "output_blocks.7.", "output_blocks.8."],
    "UP3": ["output_blocks.9.", "output_blocks.10.", "output_blocks.11.", "out."]
}
block_4_keys_xl = {
    "DOWN0": ["label_emb.", "time_embed.", "input_blocks.0.", "input_blocks.1.", "input_blocks.2.", "input_blocks.3.", "input_blocks.4."],
    "DOWN1": ["input_blocks.5.", "input_blocks.6."],
    "DOWN2": ["input_blocks.7.",],
    "DOWN3": ["input_blocks.8."],
    "MID": ["middle_block."],
    "UP0": ["output_blocks.0.", "output_blocks.1.", "output_blocks.2.", "output_blocks.3.", "output_blocks.4.",],
    "UP1": ["output_blocks.5.", "output_blocks.6."],
    "UP2": ["output_blocks.7."],
    "UP3": ["output_blocks.8.", "out."]
}

block_9_labels = ["IN0","IN1", "IN2", "IN3", "IN4", "IN5", "IN6", "IN7", "IN8",
                  "M0",
                  "OUT0","OUT1", "OUT2", "OUT3", "OUT4", "OUT5", "OUT6", "OUT7", "OUT8"]

block_9_keys_xl = {
    "IN0": ["label_emb.", "time_embed.", "input_blocks.0."],
    "IN1": ["input_blocks.1."],
    "IN2": ["input_blocks.2."],
    "IN3": ["input_blocks.3."],
    "IN4": ["input_blocks.4."],
    "IN5": ["input_blocks.5."],
    "IN6": ["input_blocks.6."],
    "IN7": ["input_blocks.7."],
    "IN8": ["input_blocks.8."],
    "M0": ["middle_block."],
    "OUT0": ["output_blocks.0."],
    "OUT1": ["output_blocks.1."], 
    "OUT2": ["output_blocks.2."],
    "OUT3": ["output_blocks.3."],
    "OUT4": ["output_blocks.4."],
    "OUT5": ["output_blocks.5."],
    "OUT6": ["output_blocks.6."],
    "OUT7": ["output_blocks.7."],
    "OUT8": ["output_blocks.8.", "out."],
}

block_12_labels = [
    "IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11",
    "M00",
    "OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"
]
block_12_keys = {
    "IN00": ["time_embed.", "input_blocks.0."],
    "IN01": ["input_blocks.1."], "IN02": ["input_blocks.2."],
    "IN03": ["input_blocks.3."], "IN04": ["input_blocks.4."],
    "IN05": ["input_blocks.5."], "IN06": ["input_blocks.6."],
    "IN07": ["input_blocks.7."], "IN08": ["input_blocks.8."],
    "IN09": ["input_blocks.9."], "IN10": ["input_blocks.10."],
    "IN11": ["input_blocks.11."],
    "M00": ["middle_block."],
    "OUT00": ["output_blocks.0."], "OUT01": ["output_blocks.1."],
    "OUT02": ["output_blocks.2."], "OUT03": ["output_blocks.3."],
    "OUT04": ["output_blocks.4."], "OUT05": ["output_blocks.5."],
    "OUT06": ["output_blocks.6."], "OUT07": ["output_blocks.7."],
    "OUT08": ["output_blocks.8."], "OUT09": ["output_blocks.9."],
    "OUT10": ["output_blocks.10."],
    "OUT11": ["output_blocks.11.", "out."],
}

block_12_keys_xl = {
    "IN00": ["label_emb.", "time_embed.", "input_blocks.0.", "input_blocks.1.", "input_blocks.2.", "input_blocks.3.", "input_blocks.4.", "input_blocks.5.", "input_blocks.6.", "input_blocks.7.0."],
    "IN01": ["input_blocks.7.1.norm.", "input_blocks.7.1.proj_in.", "input_blocks.7.1.proj_out.", "input_blocks.7.1.transformer_blocks.0.", "input_blocks.7.1.transformer_blocks.1."],
    "IN02": ["input_blocks.7.1.transformer_blocks.2.", "input_blocks.7.1.transformer_blocks.3."],
    "IN03": ["input_blocks.7.1.transformer_blocks.4.", "input_blocks.7.1.transformer_blocks.5."],
    "IN04": ["input_blocks.7.1.transformer_blocks.6.", "input_blocks.7.1.transformer_blocks.7."],
    "IN05": ["input_blocks.7.1.transformer_blocks.8.", "input_blocks.7.1.transformer_blocks.9."],
    "IN06": ["input_blocks.8.0.", "input_blocks.8.1.norm.", "input_blocks.8.1.proj_in.", "input_blocks.8.1.proj_out.", "input_blocks.8.1.transformer_blocks.0."],
    "IN07": ["input_blocks.8.1.transformer_blocks.1.", "input_blocks.8.1.transformer_blocks.2."],
    "IN08": ["input_blocks.8.1.transformer_blocks.3.", "input_blocks.8.1.transformer_blocks.4."],
    "IN09": ["input_blocks.8.1.transformer_blocks.5.", "input_blocks.8.1.transformer_blocks.6."],
    "IN10": ["input_blocks.8.1.transformer_blocks.7.", "input_blocks.8.1.transformer_blocks.8."],
    "IN11": ["input_blocks.8.1.transformer_blocks.9.", "middle_block.0."],
    "M00": ["middle_block.1."],
    "OUT00": ["middle_block.2.", "output_blocks.0.0."],
    "OUT01": ["output_blocks.0.1.norm.", "output_blocks.0.1.proj_in.", "output_blocks.0.1.proj_out.", "output_blocks.0.1.transformer_blocks.0.", "output_blocks.0.1.transformer_blocks.1."],
    "OUT02": ["output_blocks.0.1.transformer_blocks.2.", "output_blocks.0.1.transformer_blocks.3.", "output_blocks.0.1.transformer_blocks.4.", "output_blocks.0.1.transformer_blocks.5."],
    "OUT03": ["output_blocks.0.1.transformer_blocks.6.", "output_blocks.0.1.transformer_blocks.7.", "output_blocks.0.1.transformer_blocks.8.", "output_blocks.0.1.transformer_blocks.9."],
    "OUT04": ["output_blocks.1.0.", "output_blocks.1.1.norm.", "output_blocks.1.1.proj_in.", "output_blocks.1.1.proj_out.", "output_blocks.1.1.transformer_blocks.0.", "output_blocks.1.1.transformer_blocks.1."],
    "OUT05": ["output_blocks.1.1.transformer_blocks.2.", "output_blocks.1.1.transformer_blocks.3.", "output_blocks.1.1.transformer_blocks.4.", "output_blocks.1.1.transformer_blocks.5."],
    "OUT06": ["output_blocks.1.1.transformer_blocks.6.", "output_blocks.1.1.transformer_blocks.7.", "output_blocks.1.1.transformer_blocks.8.", "output_blocks.1.1.transformer_blocks.9."],
    "OUT07": ["output_blocks.2.0.", "output_blocks.2.1.norm.", "output_blocks.2.1.proj_in.", "output_blocks.2.1.proj_out.", "output_blocks.2.1.transformer_blocks.0.", "output_blocks.2.1.transformer_blocks.1."],
    "OUT08": ["output_blocks.2.1.transformer_blocks.2.", "output_blocks.2.1.transformer_blocks.3.", "output_blocks.2.1.transformer_blocks.4."],
    "OUT09": ["output_blocks.2.1.transformer_blocks.5.", "output_blocks.2.1.transformer_blocks.6.", "output_blocks.2.1.transformer_blocks.7."],
    "OUT10": ["output_blocks.2.1.transformer_blocks.8.", "output_blocks.2.1.transformer_blocks.9.", "output_blocks.2.2.", "output_blocks.3."],
    "OUT11": ["output_blocks.4.", "output_blocks.5.", "output_blocks.6.", "output_blocks.7.", "output_blocks.8.", "out."]
}

def base64_encode(obj):
    return base64.b64encode(str(obj).encode('utf-8')).decode('utf-8')

def weighted_sum(a, b, alpha):
    return (alpha * a) + ((1-alpha) * b)

def add_difference(a, b, c, alpha):
    return a + alpha*(b - c)

def add(a, b, alpha):
    return a + alpha*b

def difference(a, b, alpha):
    return alpha*(a - b)

def weighted(a, alpha):
    return alpha * a

def combine(a, b, alpha):
    return a + b

def get_result_history(recipe, result_name):
    index = int(result_name.split("_")[-1])
    full_op = recipe[index]

    important_keys = ["operation", "alpha", "model_a", "model_b", "model_c", "clip_alpha", "rank", "conv_rank"]
    operation = {k:full_op[k] for k in important_keys if k in full_op}

    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if operation[k].startswith("_result_"):
            operation[k] = get_result_history(recipe, operation[k])
    
    return operation

def get_required_model_names(self, operation, prune=False):
    name = base64_encode(operation)
    if name in self.storage.loaded["UNET"] and prune:
        return [name]
    names = [name]
    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if type(operation[k]) == dict:
            names += get_required_model_names(self, operation[k], prune)
        else:
            names += [operation[k]]
    return names

def get_base_model_names(self, operation):
    names = []
    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if type(operation[k]) == dict:
            names += get_base_model_names(self, operation[k])
        else:
            names += [operation[k]]
    return names

def do_unet_insert_lora(self, inputs, alpha):
    base_input, lora_input = inputs

    base_state_dict = base_input.state_dict()
    lora_state_dict = lora_input.compose()

    key_mapping = None
    if type(alpha) == list:
        key_mapping = utils.block_mapping(len(alpha))

    out_state_dict = {}

    self.set_status("Merging")

    for k in base_state_dict:
        base_value = base_state_dict[k]
        if k.endswith(".weight"):
            lora_key = "lora_unet_" + k.rsplit(".",1)[0].replace(".","_")
            if lora_key in lora_state_dict:
                lora_value = lora_state_dict[lora_key].reshape(base_value.shape)
                if key_mapping:
                    out_state_dict[k] = add(base_value, lora_value, alpha[key_mapping[k]])
                else:
                    out_state_dict[k] = add(base_value, lora_value, alpha)
                continue
        out_state_dict[k] = base_value.clone()

    out_state_dict["metadata"] = {
        "model_type": base_input.model_type,
        "model_variant": base_input.model_variant,
        "prediction_type": base_input.prediction_type
    }

    out = models.UNET.from_model("", out_state_dict, self.storage.dtype)
    
    return out

def do_clip_insert_lora(self, inputs, alpha):
    base_input, lora_input = inputs

    base_state_dict = base_input.state_dict()
    lora_state_dict = lora_input.compose()

    out_state_dict = {}

    self.set_status("Merging")

    for k in base_state_dict:
        base_value = base_state_dict[k]
        if k.endswith(".weight"):
            lora_key = "lora_te_" + k.rsplit(".",1)[0].replace(".","_")
            if lora_key in lora_state_dict:
                lora_value = lora_state_dict[lora_key].reshape(base_value.shape)
                out_state_dict[k] = add(base_value, lora_value, alpha)
                continue
        out_state_dict[k] = base_value.clone()

    out_state_dict["metadata"] = {
        "model_type": base_input.model_type
    }

    out = models.CLIP.from_model("", out_state_dict, self.storage.dtype)
    
    return out

def do_unet_merge(self, inputs, alpha, merge_function):
    arch = set([(m.model_type, m.model_variant) for m in inputs])
    if len(arch) != 1:
        raise Exception("Incompatible model architectures: " + str(arch))
    arch = (inputs[0].model_type, inputs[0].model_variant, inputs[0].prediction_type)
    
    state_dicts = [m.state_dict() for m in inputs]

    key_mapping = None
    if type(alpha) == list:
        key_mapping = utils.block_mapping(len(alpha))

    out_state_dict = {}

    self.set_status("Merging")

    for k in state_dicts[0]:
        if key_mapping:
            if k in key_mapping:
                out_state_dict[k] = merge_function(*[sd[k] for sd in state_dicts], alpha[key_mapping[k]])
        else:
            out_state_dict[k] = merge_function(*[sd[k] for sd in state_dicts], alpha)

    out_state_dict["metadata"] = {
        "model_type": inputs[0].model_type,
        "model_variant": inputs[0].model_variant,
        "prediction_type": inputs[0].prediction_type
    }

    out = models.UNET.from_model("", out_state_dict, self.storage.dtype)
    
    return out

def do_clip_merge(self, inputs, alpha, merge_function):
    arch = set([m.model_type for m in inputs])
    if len(arch) != 1:
        raise Exception("Incompatible model architectures: " + str(arch))
    
    state_dicts = [m.state_dict() for m in inputs]

    out_state_dict = {}

    self.set_status("Merging")

    for k in state_dicts[0]:
        out_state_dict[k] = merge_function(*[sd[k] for sd in state_dicts], alpha)

    out_state_dict["metadata"] = {
        "model_type": inputs[0].model_type
    }

    out = models.CLIP.from_model("", out_state_dict, self.storage.dtype)
    
    return out

def do_recursive_checkpoint_merge(self, operation):
    operation_name = base64_encode(operation)

    if operation_name in self.storage.loaded["UNET"] and operation_name in self.storage.loaded["CLIP"]:
        return operation_name

    inputs = []

    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if type(operation[k]) == dict:
            name = do_recursive_checkpoint_merge(self, operation[k])
            inputs += [name]
        else:
            inputs += [operation[k]]   

    for comp in ["UNET", "CLIP"]:
        input_comps = []
        for name in inputs:
            input_comp = comp
            if "lora"+os.path.sep in name.lower():
                input_comp = "LoRA"
            input_comps += [self.storage.get_component(name, input_comp, torch.device("cpu"))]
        
        alpha = operation['alpha']
        clip_alpha = operation.get('clip_alpha', 1.0)

        if comp == "UNET":
            if operation["operation"] == "Insert LoRA":
                result = do_unet_insert_lora(self, input_comps, alpha)
            elif operation["operation"] == "Weighted Sum":
                result = do_unet_merge(self, input_comps, alpha, weighted_sum)
            elif operation["operation"] == "Add Difference":
                result = do_unet_merge(self, input_comps, alpha, add_difference)
        elif comp == "CLIP":
            if operation["operation"] == "Insert LoRA":
                result = do_clip_insert_lora(self, input_comps, clip_alpha)
            elif operation["operation"] == "Weighted Sum":
                result = do_clip_merge(self, input_comps, clip_alpha, weighted_sum)
            elif operation["operation"] == "Add Difference":
                result = do_clip_merge(self, input_comps, clip_alpha, add_difference)
        
        self.storage.add(comp, operation_name, result)

    return operation_name

def get_key_model(key):
    maps = {
        "model.diffusion_model.": "unet",
        "cond_stage_model.": "clip",
        "conditioner.": "clip",
        "first_stage_model.": "vae"
    }
    for k, v in maps.items():
        if key.startswith(k):
            return v
    return None

def get_key_block_weight(key, weights, is_xl):
    labels = None
    prefix = None
    if len(weights) == 19:
        labels = block_9_labels
        if is_xl:
            prefix = block_9_keys_xl
        else:
            raise Exception("9 Block weighting is not supported for SDv1")
    elif len(weights) == 9:
        labels = block_4_labels
        if is_xl:
            prefix = block_4_keys_xl
        else:
            prefix = block_4_keys
    else:
        labels = block_12_labels
        if is_xl:
            prefix = block_12_keys_xl
        else:
            prefix = block_12_keys

    key = key.replace("model.diffusion_model.", "")

    for i, b in enumerate(labels):
        for p in prefix[b]:
            if key.startswith(p):
                return weights[i], b
    
    raise Exception("Unknown key: " + key)

def do_disk_checkpoint_merge(self, operation, device):
    self.set_status("Parsing")

    operation_name = base64_encode(operation)
    comps = ["CLIP", "VAE", "UNET"]

    if all([operation_name in self.storage.loaded[comp] for comp in comps]):
        return operation_name
    
    for comp in comps:
        for name in list(self.storage.loaded[comp].keys()):
            self.storage.remove(comp, name)

    state_dict = {}
    dtype = torch.float16

    alpha = operation["alpha"]
    clip_alpha = operation["clip_alpha"]
    vae_source = operation["vae_source"]

    model_a = os.path.abspath(os.path.join(self.storage.path, operation["model_a"]))
    model_b = os.path.abspath(os.path.join(self.storage.path, operation["model_b"]))
    model_c = os.path.abspath(os.path.join(self.storage.path, operation["model_c"])) if "model_c" in operation else None

    if not all([not m or m.endswith(".safetensors") for m in [model_a, model_b, model_c]]):
        raise Exception("Only safetensor checkpoints are supported")

    op = operation["operation"]

    last_model = None

    with contextlib.ExitStack() as stack:
        a = stack.enter_context(safetensors.safe_open(model_a, framework="pt", device="cpu"))
        b = stack.enter_context(safetensors.safe_open(model_b, framework="pt", device="cpu"))
        c = stack.enter_context(safetensors.safe_open(model_c, framework="pt", device="cpu")) if model_c else None
        
        is_xl = "conditioner.embedders.1.model.text_projection" in a.keys()

        iter = tqdm.tqdm(list(enumerate(a.keys())))
        for i, k in iter:
            model = get_key_model(k)

            if not k in b.keys() or (c and not k in c.keys()):
                continue

            if model != last_model:
                self.set_status("Merging " + model.upper(), reset=False)
                last_model = model

            self.on_merge(iter.format_dict)

            at = a.get_tensor(k).to(torch.float32)
            bt = b.get_tensor(k).to(torch.float32)
            ct = c.get_tensor(k).to(torch.float32) if c else None

            if at.shape != bt.shape or (ct and at.shape != ct.shape):
                # for inpainting models
                if k == "model.diffusion_model.input_blocks.0.0.weight":
                    state_dict[k] = at.to(device, dtype)
                    continue

            if model == "vae":
                state_dict[k] = [at,bt,ct][vae_source].to(device, dtype)
                continue
            
            if model == "unet":
                w = alpha
                if type(alpha) == list:
                    w, _ = get_key_block_weight(k, alpha, is_xl)
            elif model == "clip":
                w = clip_alpha

            if op == "Weighted Sum":
                state_dict[k] = ((w * at) + ((1-w) * bt)).to(device, dtype)
            elif op == "Add Difference":
                state_dict[k] = (at + w*(bt - ct)).to(device, dtype)

            if i % 500 == 0:
                self.storage.do_gc()
    
    self.set_status("Converting")

    state_dict, metadata = convert.convert_checkpoint(state_dict)
    sub_state_dict = self.storage.parse_model(state_dict, metadata)
   
    for comp in comps:
        self.set_status("Loading " + comp.upper())
        comp_dtype = self.storage.vae_dtype if comp == "VAE" else self.storage.dtype
        model = self.storage.classes[comp].from_model(operation_name, sub_state_dict[comp], comp_dtype, device)
        self.storage.add(comp, operation_name, model)
        self.storage.do_gc()

    return operation_name

def merge_checkpoint(self, recipe):
    if len(recipe) == 1 and recipe[0]["operation"] in {"Weighted Sum", "Add Difference"}:
        result_name = do_disk_checkpoint_merge(self, recipe[0], self.device)
        self.vae = self.storage.get_vae(result_name, self.device)
        self.vae_name = self.merge_name + ".safetensors"
        all_lora = []
    else:
        final_result_name = f"_result_{len(recipe)-1}"

        self.set_status("Loading VAE")

        vae_sources = {}
        for i, op in enumerate(recipe):
            input_models = [op[k] for k in ["model_a", "model_b", "model_c"] if k in op]
            output_model = f"_result_{i}"
            vae_sources[output_model] = input_models[op.get("vae_source", 0)]
        
        vae_source = final_result_name
        while vae_source in vae_sources:
            vae_source = vae_sources[vae_source]
        
        self.vae = self.storage.get_vae(vae_source, self.device)
        self.vae_name = vae_source

        self.set_status("Loading Sources")

        self.storage.uncap_ram = True
        
        op = get_result_history(recipe, final_result_name)
        result_name = base64_encode(op)

        all = get_required_model_names(self, op, prune=False)
        required = get_required_model_names(self, op, prune=True)

        all_lora = [r for r in all if "lora"+os.path.sep in r.lower()]
        required_lora = [r for r in required if "lora"+os.path.sep in r.lower()]

        required = [r for r in required if not r in required]

        for comp in ["UNET", "CLIP"]:
            useless = []
            for name in self.storage.loaded[comp]:
                if name in required or (name in all and not self.minimal_vram):
                    continue
                useless += [name]

            for name in useless:
                self.storage.remove(comp, name)

            for name in required:
                if not "." in name:
                    continue
                self.storage.get_component(name, comp, torch.device("cpu"))

        for name in required_lora:
            self.storage.get_component(name, "LoRA", torch.device("cpu"))

        self.set_status("Merging")

        do_recursive_checkpoint_merge(self, op)  

    self.storage.clear_file_cache()

    self.unet = self.storage.get_unet(result_name, self.device)
    self.unet_name = self.merge_name + ".safetensors"

    self.clip = self.storage.get_clip(result_name, self.device)
    self.clip.set_textual_inversions(self.storage.get_embeddings(self.device))
    self.clip_name = self.merge_name + ".safetensors"

    self.storage.uncap_ram = True

    return all_lora

def do_lora_merge(self, name, inputs, rank, conv_rank, alpha, clip_alpha, merge_function):
    keys = []
    for i in inputs:
        i.precompute_decomposition(self.device, self.on_decompose)
        if len(i.decomposition.keys()) > len(keys):
            keys = i.decomposition.keys()

    key_mapping = None
    if type(alpha) == list:
        key_mapping = utils.block_mapping_lora(len(alpha))

    out_state_dict = {}
    out_decomposed = {}

    self.set_status("Merging")

    for k in keys:
        if k.startswith("lora_unet"):
            if key_mapping:
                kk = k.rsplit(".", 1)[0]
                if kk in key_mapping:
                    key_weight = alpha[key_mapping[kk]]
                else:
                    print("MISSING", kk)
                    key_weight = clip_alpha
            else:
                key_weight = alpha
        elif k.startswith("lora_te"):
            key_weight = clip_alpha
        else:
            print("UNKNOWN", k)

        if True:
            data = [i.decomposition[k] for i in inputs if k in i.decomposition]
            if len(data) != len(inputs):
                U,S,Vh,size = data[0]
            else:
                data = [i.decomposition[k] for i in inputs]
                U = merge_function(*[d[0] for d in data], key_weight)
                S = merge_function(*[d[1] for d in data], key_weight)
                Vh = merge_function(*[d[2] for d in data], key_weight)
                size = data[0][3]
            out_decomposed[k] = (U,S,Vh,size)
            key_up, key_down, key_alpha = lora.LoRANetwork.get_key_at_rank(out_decomposed[k], rank, conv_rank)
        else:
            data = [lora.LoRANetwork.get_key_at_rank(i.decomposition[k], rank, conv_rank) for i in inputs]
            key_up = merge_function(*[d[0] for d in data], key_alpha)
            key_down = merge_function(*[d[1] for d in data], key_alpha)
            key_alpha = merge_function(*[d[2] for d in data], key_alpha)

        out_state_dict[k + ".lora_up.weight"] = key_up
        out_state_dict[k + ".lora_down.weight"] = key_down
        out_state_dict[k + ".alpha"] = key_alpha

    out_lora = models.LoRA(name)
    out_lora.from_state_dict(out_state_dict)
    out_lora.decomposition = out_decomposed
    
    return out_lora

def do_lora_extraction(self, name, inputs, rank, conv_rank):
    self.storage.uncap_ram = True
    a_unet = self.storage.get_component(inputs[0], "UNET", torch.device("cpu"))
    a_clip = self.storage.get_component(inputs[0], "CLIP", torch.device("cpu"))
    b_unet = self.storage.get_component(inputs[1], "UNET", torch.device("cpu"))
    b_clip = self.storage.get_component(inputs[1], "CLIP", torch.device("cpu"))
    self.storage.clear_file_cache()

    network = models.LoRA(name)

    modules = list(a_unet.additional.modules.items()) + list(a_clip.additional.modules.items())
    for name, module in modules:
        name = "lora_" + name
        size = module.original_module.weight.shape
        conv2d = (len(size) == 4)
        kernel_size = None if not conv2d else size[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        is_conv = conv2d_3x3
        if is_conv and not conv_rank:
            continue
        module_rank = conv_rank if is_conv else rank

        lora_module = lora.LoRAModule.from_module(network.net_name, name, module.original_module, module_rank, module_rank)
        network.add_module(name, lora_module)

    lora_keys = set()
    for k in network.state_dict():
        kk = k.split(".")[0]
        if not kk in lora_keys:
            lora_keys.add(kk)

    state_dict = {}

    for prefix, comps in [("lora_unet", [a_unet, b_unet]), ("lora_te", [a_clip, b_clip])]:
        for model in comps:
            model_state_dict = model.state_dict()
            for k in model_state_dict:
                if not k.endswith(".weight"):
                    continue
                kk = prefix + "_" + k.rsplit(".", 1)[0].replace(".","_")
                if kk in lora_keys:
                    if kk in state_dict:
                        state_dict[kk] -= model_state_dict[k]
                    else:
                        state_dict[kk] = model_state_dict[k]
    
    del a_unet, a_clip, b_unet, b_clip
    for name in inputs:
        self.storage.remove("UNET", name)
        self.storage.remove("CLIP", name)

    decomposition = lora.LoRANetwork.decompose(state_dict, self.device, self.on_decompose)
    state_dict = {}
    for k in decomposition:
        key_up, key_down, key_alpha = lora.LoRANetwork.get_key_at_rank(decomposition[k], rank, conv_rank)

        state_dict[k + ".lora_up.weight"] = key_up
        state_dict[k + ".lora_down.weight"] = key_down
        state_dict[k + ".alpha"] = key_alpha

    network.load_state_dict(state_dict)
    network.decomposition = decomposition

    return network

def do_recursive_lora_merge(self, operation):
    operation_name = base64_encode(operation)

    if operation_name in self.storage.loaded["LoRA"]:
        return operation_name

    inputs = []

    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if type(operation[k]) == dict:
            name = do_recursive_lora_merge(self, operation[k])
            inputs += [name]
        else:
            inputs += [operation[k]]

    if operation["operation"] != "Extract LoRA":
        inputs = [self.storage.get_component(name, "LoRA", torch.device("cpu")) for name in inputs]

    alpha = operation.get('alpha', None)
    clip_alpha = operation.get('clip_alpha', None)

    merge_function = None
    if operation["operation"] == "Weighted Sum":
        merge_function = weighted_sum
    if operation["operation"] == "Add Difference":   
        merge_function = add_difference
    if operation["operation"] in "Modify LoRA":
        merge_function = weighted
    if operation["operation"] == "Combine LoRA":
        merge_function = combine

    net_name = f"lora:{operation_name}"
    rank = operation['rank']
    conv_rank = operation['conv_rank']

    if operation["operation"] == "Extract LoRA":
        result = do_lora_extraction(self, net_name, inputs, rank, conv_rank)
    else:
        result = do_lora_merge(self, net_name, inputs, rank, conv_rank, alpha, clip_alpha, merge_function)
    
    self.storage.add("LoRA", operation_name, result)

    return operation_name

def merge_lora(self, recipe):
    
    self.set_status("Parsing")
    raise Exception("LoRA merging temporarily unavailable")

    needed_until = {}

    sources = []
    
    for i, op in enumerate(recipe):
        input_models = [op[k] for k in ["model_a", "model_b", "model_c"] if k in op]
        output_model = f"_result_{i}"
        all_models = input_models + [output_model]
        for m in all_models:
            if not m in sources and not m.startswith("_result"):
                sources += [m]
            needed_until[m] = i
    
    final_result_name = f"_result_{len(recipe)-1}"

    op = get_result_history(recipe, final_result_name)
    all = get_required_model_names(self, op, prune=False)
    required = get_required_model_names(self, op, prune=True)

    self.set_status("Loading Sources")

    useless = []
    for name in self.storage.loaded["LoRA"]:
        if name in required or (name in all and not self.minimal_vram):
            continue
        useless += [name]

    for name in useless:
        self.storage.remove("LoRA", name)

    #for name in required:
    #    if "." in name:
    #        if name.lower().startswith("lora"):
    #            self.storage.get_component(name, "LoRA", torch.device("cpu"))
    #        else:
    #            self.storage.get_component(name, "UNET", torch.device("cpu"))

    self.set_status("Merging")
    result_name = do_recursive_lora_merge(self, op)

    keep = [n for n in all if n != result_name]

    return result_name, keep