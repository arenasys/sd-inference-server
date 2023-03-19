import torch
import PIL

import attention
import storage
import wrapper
import convert

convert.autoconvert("./models/SD", "./models/TRASH")

attention.use_optimized_attention()

storage = storage.ModelStorage("../models", torch.float16, torch.float32)
params = wrapper.GenerationParameters(storage, torch.device("cuda"))

if False:
    print("TEST 1 - txt2img")

    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, highly detailed, white hair, smug, 1girl, holding big cat")
    params.set(negative_prompt="bad")
    params.set(width=384, height=384, seed=2769446625, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("1_a.png")

    params.set(hr_factor=2.0, hr_strength=0.7, hr_steps=20)
    images = params.txt2img()
    images[0].save("1_b.png")

if True:
    print("TEST 2 - img2img")

    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, 1girl, standing in rain, coat, wet")
    params.set(negative_prompt="bad, blonde, umbrella")
    params.set(width=200, height=200, seed=4132954439, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("2_a.png")

    params.set(image=[PIL.Image.open("2_a.png")])
    params.set(prompt="masterpiece, 1girl, standing in rain, coat, wet, storm, lightning")
    params.set(seed=2839558696)
    params.set(width=512, height=768, steps=30)

    images = params.img2img()
    images[0].save("2_b.png")

if False:
    print("TEST 3 - Inpainting")

    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, siting cat, outdoors, close up, full body, long grass")
    params.set(negative_prompt="bad")
    params.set(width=512, height=512, seed=1238623037, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("3_a.png")

    mask = PIL.Image.open("mask.png").convert("L")
    params.set(image=PIL.Image.open("3_a.png"), mask=mask)
    params.set(prompt="masterpiece, siting cat, outdoors, close up, full body, long grass")
    params.set(negative_prompt="bad", strength=0.9, seed=3701325301)

    images = params.img2img()
    images[0].save(f"3_b.png")

    mask = PIL.Image.open("mask2.png").convert("L")
    params.set(image=PIL.Image.open("3_b.png"), mask=mask)
    params.set(prompt="masterpiece, cat, outdoors, close up, big yellow cat eyes")
    params.set(negative_prompt="bad", strength=0.7, seed=924719124, padding=50)

    images = params.img2img()
    images[0].save(f"3_c.png")

if False:
    print("TEST 4 - Batching")

    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt=["masterpiece, highly detailed, desert, dunes, (covered), hooded, robe, wanderer, sunset, looking away, from behind",
                       "masterpiece, highly detailed, mountain top, valley, (covered), hooded, robe, wanderer, sunset, looking away, from behind"])
    params.set(negative_prompt="bad, beach, close up, (face), artist, hands")
    params.set(width=512, height=512, seed=[1347094397, 785248764], steps=20, scale=7)
    
    images = params.txt2img()
    images[0].save("4_a.png")
    images[1].save("4_b.png")

if False:
    print("TEST 5 - Embedding")
    
    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, highly detailed, 1girl, smile, sitting on train, heterochroma, pink eyes, NeoRWBY, small")
    params.set(negative_prompt="bad, cartoon")
    params.set(width=384, height=448, seed=728158606, steps=20, scale=7)
    params.set(hr_factor=1.5, hr_strength=0.7, hr_steps=20)
    images = params.txt2img()
    images[0].save("5_a.png")

if False:
    print("TEST 6 - LoRA & HN")
    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpeice, 1girl, pink hair, bunny ears, pink eyes, jacket")
    params.set(negative_prompt="")
    params.set(width=512, height=512, seed=2234117738, steps=20, scale=7)

    params.set(lora="pippa")

    images = params.txt2img()
    images[0].save(f"6_a.png")

    params.set(lora=None)
    images = params.txt2img()
    images[0].save(f"6_b.png")

    params.set(lora="pippa", hn="aamuk-36500")

    images = params.txt2img()
    images[0].save(f"6_c.png")

if False:
    print("TEST 7 - SDv2")

    params.reset()
    params.set(model="SDv2", sampler="Euler a")
    params.set(prompt="confused cat looking up")
    params.set(negative_prompt="bad")
    params.set(width=768, height=768, seed=665746805, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("7_a.png")

if False:
    print("TEST 8 - DDIM/PLMS")
    params.reset()
    params.set(model="Anything-V3", sampler="DDIM", clip_skip=2)
    params.set(prompt="masterpiece, highly detailed, white hair, smug, 1girl, sunny, beach, clouds, hat, small")
    params.set(negative_prompt="bad")
    params.set(width=512, height=512, seed=1708444674, steps=25, scale=7)
    images = params.txt2img()
    images[0].save("8_a.png")

    params.set(sampler="PLMS")
    images = params.txt2img()
    images[0].save("8_b.png")

    mask = PIL.Image.open("mask_ddim.png").convert("L")
    params.set(image=PIL.Image.open("8_a.png"), mask=mask)
    params.set(sampler="DDIM")
    params.set(prompt="sun flower, straw hat, masterpiece, highly detailed, white hair, smug, 1girl, sunny, beach, clouds, hat, small")
    params.set(strength=0.75, seed=41398214)

    images = params.img2img()
    images[0].save(f"8_c.png")    

if False:
    print("TEST 9 - HR Sampler")
    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, highly detailed, white hair, smug, 1girl, sunny, beach, clouds, small")
    params.set(negative_prompt="bad")
    params.set(width=512, height=512, seed=1400860402, steps=25, scale=7)
    images = params.txt2img()
    images[0].save("9_a.png")

    params.set(hr_factor=1.5, hr_strength=0.35, hr_steps=25, hr_upscale="Lanczos")
    images = params.txt2img()
    images[0].save("9_b.png")

    params.set(hr_factor=1.5, hr_strength=0.35, hr_steps=25, hr_sampler="DDIM", hr_upscale="Lanczos")
    images = params.txt2img()
    images[0].save("9_c.png")

if False:
    print("TEST 10 - HR Scheduling")
    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, highly detailed, [white hair:red hair:HR], smug, 1girl, sunny, beach, clouds, small")
    params.set(negative_prompt="bad")
    params.set(width=512, height=512, seed=1400860402, steps=25, scale=7)
    params.set(hr_factor=1.5, hr_strength=0.7, hr_steps=20)
    images = params.txt2img()
    images[0].save("10_a.png")