import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import random
import torch
import storage
import wrapper
import string
import time
import urllib.parse
from server import Server

model_folder = sys.argv[1]
endpoint = "127.0.0.1:28888"

password = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(8))
print("PASSWORD:", password)

if len(sys.argv) > 2:
    web = "https://arenasys.github.io/?" + urllib.parse.urlencode({'endpoint': sys.argv[2], "password": password})
    print("WEB:", web)

model_storage = storage.ModelStorage(model_folder, torch.float16, torch.float32)
params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

ip, port = endpoint.split(':')
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