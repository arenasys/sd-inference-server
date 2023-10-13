import os
import sys
import glob
import shutil

import torch
import safetensors
import safetensors.torch

import utils

def SDv1_convert(state_dict):
    mapping = {}
    with open(utils.relative_file(os.path.join("mappings", "SDv1_mapping.txt"))) as file:
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
    
    # 1x1 conv2d to linear
    for k in state_dict:
        if k.endswith(".weight") and "proj_" in k:
            state_dict[k] = state_dict[k].squeeze()

def SDv1_revert(state_dict):
    mapping = {}
    with open(utils.relative_file(os.path.join("mappings", "SDv1_mapping.txt"))) as file:
        for line in file:
            if line.split('.',1)[0] in {"model", "first_stage_model", "cond_stage_model"}:
                src, dst = line.strip().split(" TO ")
                mapping[dst] = src

    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)

    for k in state_dict:
        if k.endswith(".weight") and "proj_" in k:
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
    with open(utils.relative_file(os.path.join("mappings", "CN_mapping.txt"))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[src] = dst

    for k in list(state_dict.keys()):
        t = k
        if not t.startswith("control_model."):
            t = "control_model." + t
        if t in mapping:
            kk = mapping[t]
            state_dict[kk] = state_dict[k]
            del state_dict[k]
            if state_dict[kk].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                state_dict[kk] = state_dict[kk].to(torch.float16)
        else:
            print(k)
            del state_dict[k]
    
    return state_dict

def SDv2_revert(state_dict):
    mapping = {}
    with open(utils.relative_file(os.path.join("mappings", "SDv2_mapping.txt"))) as file:
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
    with open(utils.relative_file(os.path.join("mappings", "SDv2_mapping.txt"))) as file:
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
            print("DEL", k)
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

def SDXL_Base_revert(state_dict):
    mapping = {}
    with open(utils.relative_file(os.path.join("mappings", "SDXL-Base_mapping.txt"))) as file:
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
        if k.endswith("text_projection"):
            state_dict[k] = state_dict[k].T.contiguous()

    for k in list(chunks.keys()):
        state_dict[k] = torch.cat(chunks[k])
        del chunks[k]

    return state_dict

def SDXL_Base_convert(state_dict):
    mapping = {}
    with open(utils.relative_file(os.path.join("mappings", "SDXL-Base_mapping.txt"))) as file:
        for line in file:
            src, dst = line.strip().split(" TO ")
            mapping[src] = dst
    
    for k in list(state_dict.keys()):
        if "attn.in_proj" in k:
            chunk = torch.chunk(state_dict[k], 3)
            state_dict["chunk0-"+k] = chunk[0]
            state_dict["chunk1-"+k] = chunk[1]
            state_dict["chunk2-"+k] = chunk[2]
            del state_dict[k]
        if k.endswith("text_projection"):
            state_dict[k] = state_dict[k].T

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

    for k in state_dict:
        if ".VAE." in k and k.endswith(".weight") and "mid_block.attentions.0." in k:
            state_dict[k] = state_dict[k].squeeze()
    
    position_ids = torch.Tensor([list(range(77))]).to(torch.int64)
    state_dict["SDXL-Base.CLIP.open_clip.text_model.embeddings.position_ids"] = position_ids
    state_dict["SDXL-Base.CLIP.ldm_clip.text_model.embeddings.position_ids"] = position_ids

    return state_dict

def clean_component(state_dict):
    valid = set()
    with open(utils.relative_file(os.path.join("mappings", "COMP_valid.txt"))) as file:
        for line in file:
            valid.add(line.strip())
    
    deleted = False
    for k in list(state_dict.keys()):
        if not k in valid:
            print("DEL", k)
            del state_dict[k]
            deleted = True
    return deleted

def convert_checkpoint(in_file):
    print(f"CONVERTING {in_file.rsplit(os.path.sep,1)[-1]}")

    metadata = {}
    name = in_file.split(os.path.sep)[-1].split(".")[0]
    if in_file.endswith(".ckpt") or in_file.endswith(".pt"):
        state_dict = utils.load_pickle(in_file, map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    elif in_file.endswith(".safetensors"):
        state_dict = {}
        with safetensors.safe_open(in_file, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            state_dict = {}
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    # GUESS MISSING MODEL INFORMATION
    if not "model_type" in metadata:
        metadata["model_type"] = "SDv1"
        if "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_bias" in state_dict:
            metadata["model_type"] = "SDv2"
        if "conditioner.embedders.1.model.text_projection" in state_dict:
            metadata["model_type"] = "SDXL-Base"
    
    if not "model_variant" in metadata:
        metadata["model_variant"] = ""
        if "model.diffusion_model.input_blocks.0.0.weight" in state_dict:
            if state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1] == 9:
                metadata["model_variant"] = "Inpainting"

    if not "prediction_type" in metadata:
        metadata["prediction_type"] = "unknown"
        yaml_file = os.path.join(os.path.dirname(in_file), name + ".yaml")
        if os.path.exists(yaml_file):
            import yaml
            with open(yaml_file, "r", encoding='utf-8') as f:
                metadata["prediction_type"] = yaml.safe_load(f)["model"]["params"].get("parameterization", "epsilon")
        else:
            if metadata["model_type"] in {"SDXL-Base"} :
                metadata["prediction_type"] = "epsilon"
                print("USING", metadata["prediction_type"], "PREDICTION")

    if metadata["model_type"] == "SDv1":
        print("CONVERTING FROM SDv1")
        SDv1_convert(state_dict)
    elif metadata["model_type"] == "SDv2":
        print("CONVERTING FROM SDv2")
        SDv2_convert(state_dict)
    elif metadata["model_type"] == "SDXL-Base":
        print("CONVERTING FROM SDXL-Base")
        SDXL_Base_convert(state_dict)        

    print("DONE")

    return state_dict, metadata

def convert_checkpoint_save(in_file, out_folder):
    name = in_file.split(os.path.sep)[-1].split(".")[0]
    state_dict, metadata = convert_checkpoint(in_file)
    
    out_file = os.path.join(out_folder, f"{name}.qst")
    print(f"SAVING {out_file}")
    safetensors.torch.save_file(state_dict, out_file, metadata)

def convert_diffusers_folder(in_folder):
    print(f"CONVERTING {in_folder.rsplit(os.path.sep,1)[-1]}")
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel
    import json

    unet_path = os.path.join(in_folder, "unet")
    vae_path = os.path.join(in_folder, "vae")
    clip_path = os.path.join(in_folder, "text_encoder")
    scheduler_file = os.path.join(in_folder, "scheduler", "scheduler_config.json")

    with open(scheduler_file, "r", encoding='utf-8') as f:
        prediction_type = json.load(f)["prediction_type"]

    unet = UNet2DConditionModel.from_pretrained(unet_path)
    model_type = "SDv2" if unet.config.cross_attention_dim == 1024 else "SDv1"

    metadata = {
        "model_type": model_type,
        "prediction_type": prediction_type,
        "model_variant": ""
    }

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

    print("DONE")

    return state_dict, metadata

def convert_diffusers_folder_save(in_folder, out_folder):
    name = in_folder.strip(os.path.sep).split(os.path.sep)[-1]
    state_dict, metadata = convert_diffusers_folder(in_folder)

    out_file = os.path.join(out_folder, f"{name}.qst")
    print(f"SAVING {out_file}")
    safetensors.torch.save_file(state_dict, out_file, metadata)

def convert(model_path):
    if os.path.isfile(model_path):
        return convert_checkpoint(model_path)
    elif os.path.isdir(model_path):
        return convert_diffusers_folder(model_path)
    else:
        raise ValueError(f"invalid model: {model_path}")

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
            convert_diffusers_folder_save(model, folder)
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
    elif model_type == "SDXL-Base":
        return SDXL_Base_revert(state_dict)
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
        convert_diffusers_folder_save(in_path, out_folder)
    else:
        convert_checkpoint_save(in_path, out_folder)