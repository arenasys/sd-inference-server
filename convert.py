import os
import sys
import glob
import shutil

import torch
import safetensors.torch

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

    # use_linear_projection: False
    for k in state_dict:
        if ".UNET." in k and k.endswith(".weight") and "proj_" in k:
            state_dict[k] = state_dict[k].unsqueeze(2).unsqueeze(2)

    # SDv2 models dont come with position IDs (probably because they should never be changed)
    position_ids = torch.Tensor([list(range(77))]).to(torch.int64)
    state_dict["SDv2.CLIP.text_model.embeddings.position_ids"] = position_ids

    return state_dict

def convert_checkpoint(in_file, out_folder):
    print(f"CONVERTING {in_file}")

    name = in_file.split(os.path.sep)[-1].split(".")[0]
    if in_file.endswith(".ckpt"):
        state_dict = torch.load(in_file, map_location="cpu")["state_dict"]
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
            with open(yaml_file, "r") as f:
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

    with open(scheduler_file, "r") as f:
        prediction_type = json.load(f)["prediction_type"]

    unet = UNet2DConditionModel.from_pretrained(unet_path)
    model_type = "SDv2" if unet.cross_attention_dim == 1024 else "SDv1"

    print("CONVERTING FROM", model_type)

    state_dict = {}
    unet = unet.to(torch.float16).state_dict()
    for k in list(unet.keys()):
        state_dict[f"{model_type}.UNET.{k}"] = unet[k]
        del unet[k]

    # use_linear_projection: False
    for k in state_dict:
        if ".UNET." in k and k.endswith(".weight") and "proj_" in k:
            state_dict[k] = state_dict[k].unsqueeze(2).unsqueeze(2)

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
            convert_checkpoint(checkpoint, folder)
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
        convert_checkpoint(in_path, out_folder)