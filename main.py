import torch
import PIL

import attention
import storage
import wrapper

attention.use_split_attention()

storage = storage.ModelStorage("./models", torch.float16, torch.float32)
params = wrapper.GenerationParameters(storage, torch.device("cuda"))

if True:
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

if False:
    print("TEST 2 - img2img")

    params.reset()
    params.set(model="Anything-V3", sampler="Euler a", clip_skip=2)
    params.set(prompt="masterpiece, 1girl, standing in rain, coat, wet")
    params.set(negative_prompt="bad, blonde, umbrella")
    params.set(width=512, height=512, seed=4132954439, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("2_a.png")

    params.set(image=images[0])
    params.set(prompt="masterpiece, 1girl, standing in rain, coat, wet, storm, lightning")
    params.set(seed=2839558696)

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
    print("TEST 5 - SDv2")

    params.reset()
    params.set(model="sd-v2", sampler="Euler a")
    params.set(prompt="confused cat looking up")
    params.set(negative_prompt="bad")
    params.set(width=768, height=768, seed=665746805, steps=20, scale=7)

    images = params.txt2img()
    images[0].save("5_a.png")
