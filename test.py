import torch
import PIL.Image
import io
import bson

import websocket as ws_client
import bson

import attention
import storage
import wrapper
from server import Server

attention.use_optimized_attention()

model_storage = storage.ModelStorage("./models", torch.float16, torch.float32)
params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

server = Server(params, "127.0.0.1", "28888")
server.start()

client = ws_client.WebSocket()
client.connect("ws://127.0.0.1:28888")

request = {"type":"txt2img", "data": {
    "model":"Anything-V3", "sampler":"Euler a", "clip_skip":2,
    "prompt":"masterpiece, highly detailed, white hair, smug, 1girl, holding big cat",
    "negative_prompt":"bad", "width":384, "height":384, "seed":2769446625, "steps":20, "scale":7,
    "hr_factor":2.0, "hr_strength":0.7, "hr_steps":20
}}

client.send_binary(bson.dumps(request))

image = None

while not image:
    response = client.recv()
    response = bson.loads(response)
    
    if response["type"] == "result":
        for i, image_data in enumerate(response["data"]["images"]):
            image = PIL.Image.open(io.BytesIO(image_data))
        response["data"] = "..."
    print(response)

server.stop()

display(image)
