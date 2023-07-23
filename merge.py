
import torch
import base64
import os
import tqdm

import models
import utils
import storage

def base64_encode(obj):
    return base64.b64encode(str(obj).encode('utf-8')).decode('utf-8')

def relative_file(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

def block_mapping(labels):
    file = "MBW_4_mapping.txt" if labels == 9 else "MBW_12_mapping.txt"
    mapping = {}
    labels = []
    with open(relative_file(os.path.join("mappings", file))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            if not dst in labels:
                labels += [dst]
            mapping[src] = labels.index(dst)
    return mapping

def weighted_sum(a, b, alpha):
    return (alpha * a) + ((1-alpha) * b)

def add_difference(a, b, c, alpha):
    return a + alpha*(b - c)

def do_checkpoint_merge(self, inputs, alpha, merge_function):
    arch = set([(m.model_type, m.model_variant) for m in inputs])
    if len(arch) != 1:
        raise Exception("Incompatible model architectures: " + str(arch))
    arch = (inputs[0].model_type, inputs[0].model_variant, inputs[0].prediction_type)
    
    state_dicts = [m.state_dict() for m in inputs]

    key_mapping = None
    if type(alpha) == list:
        key_mapping = block_mapping(len(alpha))

    out_state_dict = {}

    iter = tqdm.tqdm(state_dicts[0])
    for k in iter:
        self.on_merge(iter.format_dict)
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

def do_lora_merge(self, name, state_dicts, alpha, merge_function):
    arch = set([tuple(sorted(i.keys())) for i in state_dicts])
    if len(arch) != 1:
        raise Exception("Incompatible model architectures")
    keys = arch.pop()

    key_mapping = None
    if type(alpha) == list:
        key_mapping = block_mapping(len(alpha))

    out_state_dict = {}

    iter = tqdm.tqdm(keys)
    for k in iter:
        self.on_merge(iter.format_dict)
        if key_mapping:
            if k in key_mapping:
                out_state_dict[k] = merge_function(*[sd[k] for sd in state_dicts], alpha[key_mapping[k]])
        else:
            out_state_dict[k] = merge_function(*[sd[k] for sd in state_dicts], alpha)
    
    if self.network_mode == "Static":
        for k in list(out_state_dict.keys()):
            out_state_dict[k+".weight"] = out_state_dict[k]
            del out_state_dict[k]
        return models.LoRA(name, out_state_dict, composed=True)
    else:
        out_state_dict = models.LoRA.decompose(out_state_dict, 64, 64, self.on_decompose)
        return models.LoRA(name, out_state_dict)

def get_result_history(recipe, result_name):
    index = int(result_name.split("_")[-1])
    full_op = recipe[index]
    operation = {k:full_op[k] for k in ["operation", "alpha", "model_a", "model_b", "model_c"] if k in full_op}

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

def do_recursive_merge(self, operation, comp="UNET"):
    operation_name = base64_encode(operation)
    if comp == "LoRA":
        operation_name += self.network_mode

    if operation_name in self.storage.loaded[comp]:
        return operation_name

    inputs = []

    for k in ["model_a", "model_b", "model_c"]:
        if not k in operation:
            continue
        if type(operation[k]) == dict:
            name = do_recursive_merge(self, operation[k])
            inputs += [name]
        else:
            inputs += [operation[k]]

    inputs = [self.storage.get_component(name, comp, torch.device("cpu")) for name in inputs]

    alpha = operation['alpha']

    merge_function = None
    if operation["operation"] == "Weighted Sum":
        merge_function = weighted_sum
    if operation["operation"] == "Add Difference":   
        merge_function = add_difference

    if comp == "UNET":
        result = do_checkpoint_merge(self, inputs, alpha, merge_function)
    elif comp == "LoRA":
        inputs = [i.compose() for i in inputs]
        net_name = f"{operation_name}.safetensors"
        result = do_lora_merge(self, net_name, inputs, alpha, merge_function)
    
    self.storage.add(comp, operation_name, result)

    return operation_name

def merge_checkpoint(self, recipe, unet_nets, clip_nets):
    self.set_status("Parsing")

    needed_until = {}

    unet_sources = []
    vae_sources = {}
    clip_sources = {}
    
    for i, op in enumerate(recipe):
        input_models = [op[k] for k in ["model_a", "model_b", "model_c"] if k in op]
        output_model = f"_result_{i}"
        all_models = input_models + [output_model]
        for m in all_models:
            if not m in unet_sources and not m.startswith("_result"):
                unet_sources += [m]
            needed_until[m] = i

        vae_sources[output_model] = input_models[op["vae_source"]]
        clip_sources[output_model] = input_models[op["clip_source"]]
    
    final_result_name = f"_result_{len(recipe)-1}"

    vae_source = final_result_name
    while vae_source in vae_sources:
        vae_source = vae_sources[vae_source]

    clip_source = final_result_name
    while clip_source in clip_sources:
        clip_source = clip_sources[clip_source]

    self.storage.uncap_ram = True
    
    op = get_result_history(recipe, final_result_name)
    result_name = base64_encode(op)

    all = get_required_model_names(self, op, prune=False)
    required = get_required_model_names(self, op, prune=True)

    for comp in ["UNET", "CLIP"]:
        for name in list(self.storage.loaded[comp].keys()):
            if name in required:
                if name == result_name:
                    self.storage.check_attached_networks(name, comp, unet_nets)
                else:
                    self.storage.check_attached_networks(name, comp, {})

    self.set_status("Loading CLIP")
    self.clip = self.storage.get_clip(clip_source, self.device, clip_nets)
    self.clip_name = clip_source
    self.clip.set_textual_inversions(self.storage.get_embeddings(self.device))

    self.set_status("Loading VAE")
    self.vae = self.storage.get_vae(vae_source, self.device)
    self.vae_name = vae_source

    self.set_status("Loading Models")

    useless = []
    for name in self.storage.loaded["UNET"]:
        if name in required or name in all:
            continue
        useless += [name]

    for name in useless:
        self.storage.remove("UNET", name)

    for name in required:
        if "." in name:
            self.storage.get_component(name, "UNET", torch.device("cpu"))

    self.set_status("Merging")
    do_recursive_merge(self, op, "UNET")

    self.storage.clear_file_cache()
    self.unet = self.storage.get_unet(result_name, self.device, unet_nets)
    self.unet_name = self.merge_name + ".safetensors"

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

    self.storage.uncap_ram = True
    
    op = get_result_history(recipe, final_result_name)
    all = get_required_model_names(self, op, prune=False)
    required = get_required_model_names(self, op, prune=True)

    self.set_status("Loading LoRAs")

    useless = []
    for name in self.storage.loaded["LoRA"]:
        if name in required or name in all:
            continue
        useless += [name]

    for name in useless:
        self.storage.remove("LoRA", name)

    for name in required:
        if "." in name:
            self.storage.get_component(name, "LoRA", torch.device("cpu"))

    self.set_status("Merging")
    result_name = do_recursive_merge(self, op, "LoRA")

    return result_name