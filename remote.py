import os
import sys

venv = os.path.abspath(os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages"))
if not sys.path[-1] == venv:
    sys.path.append(venv)

import random
import torch
import storage
import wrapper
import string
import time
from server import Server

model_folder = sys.argv[1]

model_storage = storage.ModelStorage(model_folder, torch.float16, torch.float32)
params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

password = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(8))

server = Server(params, "127.0.0.1", "28888", password)
server.start()

print("PASSWORD: ", password)

try:
    while True:
        time.sleep(1)
except:
    server.stop()
    #try_cloudflare.terminate(28888)