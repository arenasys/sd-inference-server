import os
import sys

import torch
import safetensors.torch

def relative_file(file):
    return os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), file)

def SDv1_convert(state_dict):
    mapping = {}
    with open(relative_file("SDv1_mapping.txt")) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[src] = dst

    # fix NAI bullshit
    NAI = {
        'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
        'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
        'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.'
    }
    for k in list(state_dict.keys()):
        for r in NAI:
            if k.startswith(r):
                state_dict[k.replace(r, NAI[r])] = state_dict[k]
                del state_dict[k]
                break

    # remove unknown keys
    for k in list(state_dict.keys()):
        if not k in mapping:
            del state_dict[k]

    # cast to fp16 💢
    for k in state_dict:
        if state_dict[k].dtype == torch.float32:
            state_dict[k] = state_dict[k].to(torch.float16)

    # most keys are just renamed
    for src, dst in mapping.items():
        state_dict[dst] = state_dict[src]
        del state_dict[src]
    
    # some keys in the VAE need to be squeezed
    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k:
            state_dict[k] = state_dict[k].squeeze()

def SDv2_convert(state_dict):
    mapping = {}
    with open(relative_file("SDv2_mapping.txt")) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[src] = dst

    # some keys in CLIP need to be chunked (then renamed later)
    for k in list(state_dict.keys()):
        if "attn.in_proj" in k:
            chunk = torch.chunk(state_dict[k], 3)
            state_dict["chunk0-"+k] = chunk[0]
            state_dict["chunk1-"+k] = chunk[1]
            state_dict["chunk2-"+k] = chunk[2]
            del state_dict[k]

    # remove unknown keys
    for k in list(state_dict.keys()):
        if not k in mapping:
            del state_dict[k]

    # cast to fp16
    for k in state_dict:
        if state_dict[k].dtype == torch.float32:
            state_dict[k] = state_dict[k].to(torch.float16)

    # most keys are just renamed
    for src, dst in mapping.items():
        state_dict[dst] = state_dict[src]
        del state_dict[src]

    # some keys in the VAE need to be squeezed
    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k:
            state_dict[k] = state_dict[k].squeeze()

    # some keys in the UNET need to be unsqueezed
    for k in state_dict:
        if ".UNET." in k and k.endswith(".weight") and "proj_" in k:
            state_dict[k] = state_dict[k].unsqueeze(2).unsqueeze(2)

    # SDv2 models dont come with position IDs (probably because they should never be changed)
    position_ids = torch.Tensor([list(range(77))]).to(torch.int64)
    state_dict["SDv2.CLIP.text_model.embeddings.position_ids"] = position_ids

    return state_dict

def convert(in_file, out_folder):
    print(f"LOADING {in_file}")
    name = in_file.split(".")[0].split(os.path.sep)[-1]
    if in_file.endswith(".ckpt"):
        state_dict = torch.load(in_file, map_location="cpu")["state_dict"]
    elif in_file.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(in_file, "cpu")

    # only in the SDv2 CLIP
    v2 = "cond_stage_model.model.text_projection" in state_dict

    if v2:
        print("CONVERTING FROM SDv2")
        SDv2_convert(state_dict)
    else:
        print("CONVERTING FROM SDv1")
        SDv1_convert(state_dict)

    model_type = "SDv2" if v2 else "SDv1"
    state_dict["metadata.model_type"] = torch.as_tensor([ord(c) for c in model_type])
    
    out_file = os.path.join(out_folder, f"{name}.st")
    print(f"SAVING {out_file}")
    safetensors.torch.save_file(state_dict, out_file)
    
def compare(a, b):
    a = safetensors.torch.load_file(a, "cpu")
    b = safetensors.torch.load_file(b, "cpu")

    print(len(a.keys()), len(b.keys()))

    for k in a:
        if a[k].shape != b[k].shape:
            print(k, a[k].shape, b[k].shape)
        elif not torch.equal(a[k], b[k]):
            print(k, a[k], b[k])
    
    for k in b:
        if not k in a:
            print(k)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model conversion')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('folder', type=str, help='folder to save model')
    
    args = parser.parse_args()
    in_file = args.model
    out_folder = args.folder

    convert(in_file, out_folder)