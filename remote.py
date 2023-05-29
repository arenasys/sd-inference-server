import os
import sys

venv = os.path.abspath(os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages"))
sys.path = [os.cwd(), venv] + [p for p in sys.path if not "conda" in p]

import random
import torch
import storage
import wrapper
import string
import time
from server import Server

model_folder = sys.argv[1]
endpoint = "127.0.0.1:28888"

model_storage = storage.ModelStorage(model_folder, torch.float16, torch.float32)
params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

password = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(8))

ip, port = endpoint.split(':')
server = Server(params, ip, port, password)
server.start()

print("ENDPOINT:", endpoint)
print("PASSWORD:", password)

try:
    try:
        while True:
            time.sleep(1)
    except:
        server.stop()
    time.sleep(1)
except:
    pass