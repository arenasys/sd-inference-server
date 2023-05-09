import os
import sys
import glob
import shutil

import torch
import safetensors.torch

import psutil
import platform
IS_WIN = platform.system() == 'Windows'

def has_handle(fpath):
    if IS_WIN:
        return False
    
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False

def relative_file(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

def SDv1_convert(state_dict):
    mapping = {}
    with open(relative_file(os.path.join("mappings", "SDv1_mapping.txt"))) as file:
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

    # cast to fp16
    for k in state_dict:
        if state_dict[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
            state_dict[k] = state_dict[k].to(torch.float16)

    # most keys are just renamed
    for src, dst in mapping.items():
        if src in state_dict:
            state_dict[dst] = state_dict[src]
            del state_dict[src]
    # some keys in the VAE need to be squeezed
    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k:
            state_dict[k] = state_dict[k].squeeze()

def SDv1_revert(state_dict):
    mapping = {}
    with open(relative_file(os.path.join("mappings", "SDv1_mapping.txt"))) as file:
        for line in file:
            if line.split('.',1)[0] in {"model", "first_stage_model", "cond_stage_model"}:
                src, dst = line.strip().split(" TO ")
                mapping[dst] = src

    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)

    for k in list(state_dict.keys()):
        if not k in mapping:
            del state_dict[k]

    for src, dst in mapping.items():
        if src in state_dict:
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    return state_dict

def CN_convert(state_dict):
    mapping = {}
    with open(relative_file(os.path.join("mappings", "CN_mapping.txt"))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[src] = dst

    for k in list(state_dict.keys()):
        if not k in mapping:
            del state_dict[k]

    for k in state_dict:
        if state_dict[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
            state_dict[k] = state_dict[k].to(torch.float16)

    for src, dst in mapping.items():
        if src in state_dict:
            state_dict[dst] = state_dict[src]
            del state_dict[src]
    
    return state_dict

def SDv2_revert(state_dict):
    mapping = {}
    with open(relative_file(os.path.join("mappings", "SDv2_mapping.txt"))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[dst] = src

    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)

    for k in list(state_dict.keys()):
        if not k in mapping:
            del state_dict[k]

    for src, dst in mapping.items():
        if src in state_dict:
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    chunks = {}
    for k in list(state_dict.keys()):
        if k.startswith("chunk"):
            c, o = k.split("-",1)
            c = int(c[-1])
            if not o in chunks:
                chunks[o] = [None,None,None]
            chunks[o][c] = state_dict[k]
            del state_dict[k]

    for k in list(chunks.keys()):
        state_dict[k] = torch.cat(chunks[k])
        del chunks[k]

    return state_dict

def SDv2_convert(state_dict):
    mapping = {}
    with open(relative_file(os.path.join("mappings", "SDv2_mapping.txt"))) as file:
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
        if state_dict[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
            state_dict[k] = state_dict[k].to(torch.float16)

    # most keys are just renamed
    for src, dst in mapping.items():
        if src in state_dict:
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    # some keys in the VAE need to be squeezed
    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k:
            state_dict[k] = state_dict[k].squeeze()
    
    # SDv2 models dont come with position IDs (probably because they should never be changed)
    position_ids = torch.Tensor([list(range(77))]).to(torch.int64)
    state_dict["SDv2.CLIP.text_model.embeddings.position_ids"] = position_ids

    return state_dict

def convert_checkpoint(in_file):
    print(f"CONVERTING {in_file.rsplit(os.path.sep,1)[-1]}")
    
    if has_handle(os.path.abspath(in_file)):
        raise Exception("model is still being written")

    name = in_file.split(os.path.sep)[-1].split(".")[0]
    if in_file.endswith(".ckpt") or in_file.endswith(".pt"):
        state_dict = torch.load(in_file, map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    elif in_file.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(in_file, "cpu")

    # only in the SDv2 CLIP
    v2 = "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_bias" in state_dict
    prediction_type = "epsilon"

    if v2:
        print("CONVERTING FROM SDv2")
        yaml_file = os.path.join(os.path.dirname(in_file), name + ".yaml")
        if os.path.exists(yaml_file):
            import yaml
            with open(yaml_file, "r", encoding='utf-8') as f:
                prediction_type = yaml.safe_load(f)["model"]["params"].get("parameterization", "epsilon")
        else:
            print("NO YAML FOUND, ASSUMING", prediction_type, "PREDICTION")

        SDv2_convert(state_dict)
    else:
        print("CONVERTING FROM SDv1")
        SDv1_convert(state_dict)

    model_type = "SDv2" if v2 else "SDv1"
    state_dict["metadata.model_type"] = torch.as_tensor([ord(c) for c in model_type])
    state_dict["metadata.prediction_type"] = torch.as_tensor([ord(c) for c in prediction_type])
    
    return state_dict

def convert_checkpoint_save(in_file, out_folder):
    name = in_file.split(os.path.sep)[-1].split(".")[0]
    state_dict = convert_checkpoint(in_file)
    
    out_file = os.path.join(out_folder, f"{name}.st")
    print(f"SAVING {out_file}")
    safetensors.torch.save_file(state_dict, out_file)

def convert_diffusers_folder(in_folder, out_folder):
    print(f"CONVERTING {in_folder}")
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel
    import json

    name = in_folder.strip(os.path.sep).split(os.path.sep)[-1]
    unet_path = os.path.join(in_folder, "unet")
    vae_path = os.path.join(in_folder, "vae")
    clip_path = os.path.join(in_folder, "text_encoder")
    scheduler_file = os.path.join(in_folder, "scheduler", "scheduler_config.json")

    with open(scheduler_file, "r", encoding='utf-8') as f:
        prediction_type = json.load(f)["prediction_type"]

    unet = UNet2DConditionModel.from_pretrained(unet_path)
    model_type = "SDv2" if unet.cross_attention_dim == 1024 else "SDv1"

    print("CONVERTING FROM", model_type)

    state_dict = {}
    unet = unet.to(torch.float16).state_dict()
    for k in list(unet.keys()):
        state_dict[f"{model_type}.UNET.{k}"] = unet[k]
        del unet[k]

    vae = AutoencoderKL.from_pretrained(vae_path).to(torch.float16).state_dict()
    for k in list(vae.keys()):
        state_dict[f"{model_type}.VAE.{k}"] = vae[k]
        del vae[k]

    clip = CLIPTextModel.from_pretrained(clip_path).to(torch.float16).state_dict()
    for k in list(clip.keys()):
        state_dict[f"{model_type}.CLIP.{k}"] = clip[k]
        del clip[k]

    state_dict["metadata.model_type"] = torch.as_tensor([ord(c) for c in model_type])
    state_dict["metadata.prediction_type"] = torch.as_tensor([ord(c) for c in prediction_type])

    out_file = os.path.join(out_folder, f"{name}.st")
    print(f"SAVING {out_file}")
    safetensors.torch.save_file(state_dict, out_file)

def autoconvert(folder, trash):
    checkpoints = glob.glob(os.path.join(folder, "*.ckpt"))
    checkpoints += glob.glob(os.path.join(folder, "*.safetensors"))
    for checkpoint in checkpoints:
        try:
            convert_checkpoint_save(checkpoint, folder)
        except Exception as e:
            print("FAILED", str(e))
            continue
        os.makedirs(trash, exist_ok=True)
        shutil.move(checkpoint, trash)
        yaml_file = checkpoint.rsplit(".",1)[0] + ".yaml"
        if os.path.exists(yaml_file):
            shutil.move(yaml_file, trash)

    diffusers = glob.glob(os.path.join(folder, "*/"))

    if not diffusers:
        return

    for model in diffusers:
        comps = [f.split(os.path.sep)[-1] for f in glob.glob(os.path.join(model, "*"))]
        if not "unet" in comps or not "vae" in comps or not "text_encoder" in comps or not "scheduler" in comps:
            continue
        try:
            convert_diffusers_folder(model, folder)
        except Exception as e:
            print("FAILED", str(e))
            continue
        os.makedirs(trash, exist_ok=True)
        shutil.move(model, trash)

def revert(model_type, state_dict):
    if model_type == "SDv1":
        return SDv1_revert(state_dict)
    elif model_type == "SDv2":
        return SDv2_revert(state_dict)
    else:
        raise ValueError(f"unknown model type: {model_type}")
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model conversion')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('folder', type=str, help='folder to save model')
    
    args = parser.parse_args()
    in_path = args.model
    out_folder = args.folder

    if os.path.isdir(in_path):
        convert_diffusers_folder(in_path, out_folder)
    else:
        convert_checkpoint_save(in_path, out_folder)