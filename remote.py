import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import torch
import storage
import wrapper
import string
import time
import argparse
import urllib.parse
from server import Server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', type=str, default="127.0.0.1:28888")
    parser.add_argument('--password', type=str, default="")
    parser.add_argument('--endpoint', type=str, default="")
    parser.add_argument('--models', type=str,default="./models")
    args = parser.parse_args()

    ip, port = args.bind.split(':')
    model_folder = args.models

    password = args.password
    endpoint = args.endpoint
    if endpoint:
        print("ENDPOINT:", endpoint)
    if not password:
        password = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(8))
        print("PASSWORD:", password)
    if endpoint:
        print("WEB:", "https://arenasys.github.io/?" + urllib.parse.urlencode({'endpoint': endpoint, "password": password}))

    model_storage = storage.ModelStorage(model_folder, torch.float16, torch.float32)
    params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

    server = Server(params, ip, port, password)
    server.start()

    try:
        try:
            while True:
                time.sleep(1)
        except:
            pass
        time.sleep(1)
    except:
        pass
    server.stop()