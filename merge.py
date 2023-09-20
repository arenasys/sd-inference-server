
import torch
import base64
import os
import tqdm

import models
import utils
import storage
import lora

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
        if k.endswith(".weight"):
            lora_key = "lora_unet_" + k.rsplit(".",1)[0].replace(".","_")
            if lora_key in lora_state_dict:
                if key_mapping:
                    out_state_dict[k] = add(base_state_dict[k], lora_state_dict[lora_key], alpha[key_mapping[k]])
                else:
                    out_state_dict[k] = add(base_state_dict[k], lora_state_dict[lora_key], alpha)
                continue
        out_state_dict[k] = base_state_dict[k].clone()

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
        if k.endswith(".weight"):
            lora_key = "lora_te_" + k.rsplit(".",1)[0].replace(".","_")
            if lora_key in lora_state_dict:
                out_state_dict[k] = add(base_state_dict[k], lora_state_dict[lora_key], alpha)
                continue
        out_state_dict[k] = base_state_dict[k].clone()

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

def merge_checkpoint(self, recipe):
    self.set_status("Parsing")

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
                    key_alpha = alpha[key_mapping[kk]]
                else:
                    print("MISSING", kk)
                    key_alpha = clip_alpha
            else:
                key_alpha = alpha
        else:
            key_alpha = clip_alpha

        if True:
            if any([not k in i.decomposition for i in inputs]):
                U,S,Vh,size = inputs[0].decomposition[k]
            else:
                data = [i.decomposition[k] for i in inputs]
                U = merge_function(*[d[0] for d in data], key_alpha)
                S = merge_function(*[d[1] for d in data], key_alpha)
                Vh = merge_function(*[d[2] for d in data], key_alpha)
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
    out_lora.decomposed = out_decomposed
    
    return out_lora

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

    inputs = [self.storage.get_component(name, "LoRA", torch.device("cpu")) for name in inputs]

    alpha = operation['alpha']

    merge_function = None
    if operation["operation"] == "Weighted Sum":
        merge_function = weighted_sum
    if operation["operation"] == "Add Difference":   
        merge_function = add_difference
    if operation["operation"] == "Modify LoRA":
        merge_function = weighted

    net_name = f"lora:{operation_name}"
    clip_alpha = operation['clip_alpha']
    rank = operation['rank']
    conv_rank = operation['conv_rank']
    result = do_lora_merge(self, net_name, inputs, rank, conv_rank, alpha, clip_alpha, merge_function)
    
    self.storage.add("LoRA", operation_name, result)

    return operation_name

def merge_lora(self, recipe):
    self.set_status("Parsing")

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

    self.set_status("Loading LoRAs")

    useless = []
    for name in self.storage.loaded["LoRA"]:
        if name in required or (name in all and not self.minimal_vram):
            continue
        useless += [name]

    for name in useless:
        self.storage.remove("LoRA", name)

    for name in required:
        if "." in name:
            self.storage.get_component(name, "LoRA", torch.device("cpu"))

    self.set_status("Merging")
    result_name = do_recursive_lora_merge(self, op)

    keep = [n for n in all if n != result_name]

    return result_name, keep