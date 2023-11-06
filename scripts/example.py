import PIL.Image
import argparse
from scripts import Connection, encode_image, decode_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sd-inference-server')
    parser.add_argument('--endpoint', type=str, help='endpoint address', default="ws://127.0.0.1:28888")
    parser.add_argument('--password', type=str, help='endpoint password', default="")
    args = parser.parse_args()

    con = Connection(args.endpoint, args.password)

    con.do_connect()
    opts = con.do_request({"type":"options"}, "options")["data"]
    model = opts["UNET"][0]

    request = {
        "type": "txt2img",
        "data": {
            "prompt": [[["hello world"], ["bad"]]],
            "width": 512,
            "height": 512,
            "vae": model,
            "unet": model,
            "clip": model,
            "steps": 25,
            "scale": 7.0,
            "seed": -1,
            "sampler": "Euler a"
        }
    }
    result = con.do_request(request, "result")
    result = decode_image(result["data"]["images"][0])
    result.save("result.png")

    con.do_disconnect()
