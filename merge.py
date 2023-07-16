
import torch
import base64
import os

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

def do_single_merge(self, inputs, alpha, merge_function):
    arch = set([(m.model_type, m.model_variant) for m in inputs])
    if len(arch) != 1:
        raise Exception("Incompatible model architectures: " + str(arch))
    arch = (inputs[0].model_type, inputs[0].model_variant, inputs[0].prediction_type)
    
    state_dicts = [m.state_dict() for m in inputs]

    key_mapping = None
    if type(alpha) == list:
        key_mapping = block_mapping(len(alpha))

    out_state_dict = {}
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

def do_recursive_merge(self, operation):
    operation_name = base64_encode(operation)

    if operation_name in self.storage.loaded["UNET"]:
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

    inputs = [self.storage.get_component(name, "UNET", torch.device("cpu")) for name in inputs]

    alpha = operation['alpha']

    merge_function = None
    if operation["operation"] == "Weighted Sum":
        merge_function = weighted_sum
    if operation["operation"] == "Add Difference":   
        merge_function = add_difference
        
    result = do_single_merge(self, inputs, alpha, merge_function)
    
    self.storage.add("UNET", operation_name, result)

    return operation_name

def merge(self, recipe, unet_nets, clip_nets):
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
    do_recursive_merge(self, op)

    self.storage.clear_file_cache()
    self.unet = self.storage.get_unet(result_name, self.device, unet_nets)
    self.unet_name = "Merge"