import PIL.Image
import io
import websocket as ws_client
import bson

if __name__ == "__main__":
    client = ws_client.WebSocket()
    client.connect("ws://127.0.0.1:28888")

    request = {"type":"txt2img", "data": {
        "model":"Anything-V3", "sampler":"Euler a", "clip_skip":2,
        "prompt":"masterpiece, highly detailed, white hair, smug, 1girl, holding big cat",
        "negative_prompt":"bad", "width":384, "height":384, "seed":2769446625, "steps":20, "scale":7,
        "hr_factor":2.0, "hr_strength":0.7, "hr_steps":20
    }}

    client.send_binary(bson.dumps(request))

    try:
        while True:
            response = client.recv()
            response = bson.loads(response)
            
            if response["type"] == "result":
                for i, image_data in enumerate(response["data"]["images"]):
                    image = PIL.Image.open(io.BytesIO(image_data))
                    image.save(f"client_{i}.png")
                response["data"] = "..."
            print(response)

            if False:
                abort = bson.dumps({"type": "abort", "data": {"message":"abort"}})
                client.send_binary(abort)

    except KeyboardInterrupt:
        client.close()