import threading
import traceback
import queue
import os
import torch
import sys
import datetime
import argparse

import threading
import queue
import websockets.exceptions
import websockets.sync.server
import bson
import time

import json
from urllib.parse import unquote

gofile_token = None

import attention
import storage
import wrapper

import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet, InvalidToken

import random

DEFAULT_PASSWORD = "qDiffusion"
FRAGMENT_SIZE = 1048576

def log_traceback(label):
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open("crash.log", "a") as f:
        f.write(f"{label} {datetime.datetime.now()}\n{tb}\n")
    print(label, tb)

def get_scheme(password):
    password = password.encode("utf8")
    h = hashes.Hash(hashes.SHA256())
    h.update(password)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=h.finalize()[:16], #lol
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return Fernet(key)

def get_id():
    return random.SystemRandom().randint(1, 2**31 - 1)

class Inference(threading.Thread):
    def __init__(self, wrapper, callback):
        super().__init__()
        
        self.wrapper = wrapper
        wrapper.callback = self.got_response

        self.callback = callback
        self.requests = queue.Queue()
        self.current = None

        self.uploads = {}

        self.stay_alive = True

    def got_response(self, response):
        return self.callback(self.current, response)

    def run(self):
        while self.stay_alive:
            try:
                self.current, request = self.requests.get(False)
                if request["type"] == "txt2img":
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.txt2img()
                elif request["type"] == "img2img":
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.img2img()
                elif request["type"] == "options":
                    self.wrapper.reset()
                    self.wrapper.options()
                elif request["type"] == "convert":
                    self.wrapper.reset()
                    self.wrapper.set(**request["data"])
                    self.wrapper.convert()
                elif request["type"] == "download":
                    self.download(**request["data"])
                elif request["type"] == "chunk":
                    self.upload(**request["data"])
                elif request["type"] == "ping":
                    self.got_response({"type":"pong"})
                self.requests.task_done()
            except queue.Empty:
                time.sleep(0.01)
                pass
            except Exception as e:
                if str(e) == "Aborted":
                    self.got_response({"type":"aborted", "data":{}})
                    continue
                additional = ""
                try:
                    log_traceback("SERVER")
                    s = traceback.extract_tb(e.__traceback__).format()
                    s = [e for e in s if not "venv" in e][-1]
                    s = s.split(", ")
                    file = s[0].split(os.path.sep)[-1][:-1]
                    line = s[1].split(" ")[1]
                    additional = f" ({file}:{line})"
                except Exception as a:
                    log_traceback("LOGGING")
                    additional = " THEN " + str(a)
                    pass

                self.got_response({"type":"error", "data":{"message":str(e) + additional}})
        
    def download(self, type, url):
        import gdown
        import mega
        import subprocess

        def gdownload(self, folder, url):
            try:
                parts = url.split("/")
                id = None
                if len(parts) == 4:
                    id = parts[3].split("=",1)[1].split("&",1)[0]
                elif len(parts) == 7:
                    id = parts[5]
                else:
                    raise Exception()
                gdown.download(output=folder, id=id+"&confirm=t")
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            except:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})

        def megadownload(self, folder, url):
            try:
                mega.Mega().login().download_url(url, folder)
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            except:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})

        def curldownload(self, folder, url):
            r = subprocess.run(["curl", "-I", url], capture_output=True)
            content_type = r.stdout.decode('utf-8').split("content-type: ", 1)[1].split(";",1)[0].split("\n",1)[0]
            if not content_type in {"application/octet-stream", "multipart/form-data"}:
                self.got_response({"type":"downloaded", "data":{"message": f"Unsupport type ({content_type}): " + url}})
                return
            r = subprocess.run(["curl", "-s", "-O", "-J", "-L", url], cwd=folder)
            if r.returncode == 0:
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            else:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})
        def gofiledownload(self, folder, url):
            name = unquote(url.rsplit("/",1)[-1])
            content_id = url.split("d/",1)[1].split("/",1)[0]
            global gofile_token
            if not gofile_token:
                r = subprocess.run(["curl", "-s", "https://api.gofile.io/createAccount"], capture_output=True)
                gofile_token = json.loads(r.stdout)["data"]["token"]
                subprocess.run(["curl", "-s", f"https://api.gofile.io/getAccountDetails?token={gofile_token}"], capture_output=True)
            
            r = subprocess.run(["curl", f"https://api.gofile.io/getContent?contentId={content_id}&token={gofile_token}&websiteToken=12345&cache=true"], capture_output=True)

            r = subprocess.run(["curl", "-H", f'Cookie: accountToken={gofile_token}', "-I", url], capture_output=True)
            content_type = r.stdout.decode('utf-8').split("content-type: ", 1)[1].split(";",1)[0].split("\n",1)[0]
            if content_type in {'text/plain', 'text/html'}:
                self.got_response({"type":"downloaded", "data":{"message": f"Unsupport type ({content_type}): " + url}})
                return
            r = subprocess.run(["curl", "-sH", f'Cookie: accountToken={gofile_token}', url, '-o', name], cwd=folder)
            if r.returncode == 0:
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            else:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})


        folder = os.path.join(self.wrapper.storage.path, type)

        self.got_response({"type":"downloaded", "data":{"message": "Downloading: " + url + " to " + type}})
        if not os.path.exists(folder):
            return
        if 'drive.google' in url:
            thread = threading.Thread(target=gdownload, args=([self, folder, url]))
            thread.start()
            return
        if 'mega.nz' in url:
            thread = threading.Thread(target=megadownload, args=([self, folder, url]))
            thread.start()
            return
        if 'gofile.io' in url:
            thread = threading.Thread(target=gofiledownload, args=([self, folder, url]))
            thread.start()
            return
        if 'huggingface' in url:
            url = url.replace("/blob/", "/resolve/")
        if 'civitai.com' in url and not "civitai.com/api/" in url:
            self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})
            return
        
        thread = threading.Thread(target=curldownload, args=([self, folder, url]))
        thread.start()

    def upload(self, type, name, chunk=None, index=-1):
        rel_file = os.path.join(type, name)
        file = os.path.join(self.wrapper.storage.path, rel_file)
        tmp = file + ".tmp"
        if index == 0:
            self.got_response({"type":"downloaded", "data":{"message": "Uploading: " + name + " to " + rel_file}})
            if os.path.exists(tmp):
                os.remove(tmp)
        if chunk:
            with open(tmp, 'ab') as f:
                f.write(chunk)
        else:
            if os.path.exists(tmp):
                os.rename(tmp, file)
            self.got_response({"type":"downloaded", "data":{"message": "Finished: " + rel_file}})

class Server():
    def __init__(self, wrapper, host, port, password=DEFAULT_PASSWORD):
        self.stopping = False

        self.requests = {}
        self.clients = {}

        self.scheme = None
        if password != None:
            self.scheme = get_scheme(password)

        self.inference = Inference(wrapper, callback=self.on_response)
        self.server = websockets.sync.server.serve(self.handle_connection, host=host, port=int(port), max_size=None)
        self.serve = threading.Thread(target=self.serve_forever)

    def start(self):
        print("SERVER: starting")
        self.inference.start()
        self.serve.start()

    def stop(self):
        print("SERVER: stopping")
        self.stopping = True
        self.inference.stay_alive = False
        self.server.shutdown()
        self.join()

    def join(self):
        self.serve.join()
        self.inference.join()

    def serve_forever(self):
        self.server.serve_forever()

    def handle_connection(self, connection):
        print(f"SERVER: client connected")
        client_id = get_id()
        self.clients[client_id] = queue.Queue()
        self.clients[client_id].put((-1, {"type":"hello", "data":{"id": client_id}}))
        mapping = {}
        ctr = 0
        try:
            while not self.stopping:
                if not self.clients[client_id].empty():
                    id, response = self.clients[client_id].get()
                    if id in mapping: id = mapping[id]
                    response["id"] = id
                    data = bson.dumps(response)
                    data = base64.urlsafe_b64decode(self.scheme.encrypt(data))
                    data = [data[i:min(i+FRAGMENT_SIZE,len(data))] for i in range(0, len(data), FRAGMENT_SIZE)]
                    connection.send(data)
                else:
                    data = None
                    try:
                        data = connection.recv(timeout=0)
                    except TimeoutError:
                        pass
                    if not data:
                        time.sleep(0.01)
                        ctr += 1
                        if ctr == 200:
                            connection.ping()
                            ctr = 0
                        continue
                    error = None
                    request = None
                    if type(data) in {bytes, bytearray}:
                        try:
                            if self.scheme:
                                data = self.scheme.decrypt(base64.urlsafe_b64encode(bytes(data)))
                            try:
                                request = bson.loads(data)
                            except:
                                error = "Malformed request"
                        except:
                            error = "Incorrect password"
                    else:
                        error = "Invalid request"
                    if request:
                        if request["type"] == "cancel":
                            id = 0
                            for k,v in mapping.items():
                                if v == request["data"]["id"]:
                                    id = k
                                    break
                            if id in self.requests and self.requests[id] == client_id:
                                del self.requests[id]
                                self.clients[client_id].put((id, {'type': 'aborted', 'data': {}}))

                        request_id = get_id()
                        user_id = request_id
                        if "id" in request:
                            user_id = request["id"]
                        self.requests[request_id] = client_id
                        mapping[request_id] = user_id
                        self.inference.requests.put((request_id, request))
                        self.clients[client_id].put((-1, {"type":"ack", "data":{"id": user_id}}))
                    else:
                        self.clients[client_id].put((-1, {"type":"error", "data":{"message": error}}))
        except websockets.exceptions.WebSocketException:
            del self.clients[client_id]
        except Exception as e:
            del self.clients[client_id]
            log_traceback("CLIENT")
        print(f"SERVER: client disconnected")

    def on_response(self, id, response):
        if self.stopping:
            return False
        if id in self.requests:
            client = self.requests[id]
            if client in self.clients:
                self.clients[client].put((id, response))
                return True
            else:
                return False
        else:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sd-inference-server')
    parser.add_argument('--bind', type=str, help='address (ip:port) to listen on', default="127.0.0.1:28888")
    parser.add_argument('--password', type=str, help='password to derive encryption key from', default=DEFAULT_PASSWORD)
    parser.add_argument('--models', type=str, help='models folder', default="../../models")
    args = parser.parse_args()

    ip, port = args.bind.rsplit(":",1)

    attention.use_optimized_attention()

    model_storage = storage.ModelStorage(args.models, torch.float16, torch.float32)
    params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

    server = Server(params, ip, port, args.password)
    server.start()
    
    try:
        server.join()
    except KeyboardInterrupt:
        server.stop()
