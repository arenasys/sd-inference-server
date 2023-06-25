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
    with open("crash.log", "a", encoding='utf-8') as f:
        f.write(f"{label} {datetime.datetime.now()}\n{tb}\n")
    print(label, tb)
    return tb

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


SEP = os.path.sep
INV_SEP = {"\\": '/', '/':'\\'}[os.path.sep]
NO_CONV = {"prompt", "negative_prompt", "url", "trace"}

def convert_path(p):
    return p.replace(INV_SEP, SEP)

def convert_all_paths(j):
    if type(j) == list:
        for i in range(len(j)):
            v = j[i]
            if type(v) == str and INV_SEP in v:
                j[i] = convert_path(v)
            if type(v) == list or type(v) == dict:
                convert_all_paths(j[i])
    elif type(j) == dict: 
        for k, v in j.items():
            if k in NO_CONV: continue
            if type(v) == str and INV_SEP in v:
                j[k] = convert_path(v)
            if type(v) == list or type(v) == dict:
                convert_all_paths(j[k])

class Inference(threading.Thread):
    def __init__(self, wrapper, read_only, callback):
        super().__init__()
        
        self.wrapper = wrapper
        wrapper.callback = self.got_response

        self.callback = callback
        self.requests = queue.Queue()
        self.current = None

        self.read_only = read_only
        self.owner = None

        self.uploads = {}

        self.stay_alive = True

    def got_response(self, response):
        return self.callback(self.current, response)

    def run(self):
        while self.stay_alive:
            try:
                client, self.current, request = self.requests.get(False)
                convert_all_paths(request)

                read_only = self.read_only and client != self.owner
                if read_only and request["type"] in {"convert", "manage", "download", "chunk"}:
                    raise Exception("Read-only")

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
                elif request["type"] == "manage":
                    self.wrapper.set(**request["data"])
                    self.wrapper.manage()
                elif request["type"] == "annotate":
                    self.wrapper.set(**request["data"])
                    self.wrapper.annotate()
                elif request["type"] == "segmentation":
                    self.wrapper.set(**request["data"])
                    self.wrapper.segmentation()
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
                self.requests.task_done()
                if str(e) == "Read-only":
                    self.got_response({"type":"error", "data":{"message": "Server is read-only"}})
                    continue
                if str(e) == "Aborted":
                    self.got_response({"type":"aborted", "data":{}})
                    continue
                additional = ""
                trace = ""
                try:
                    trace = log_traceback("SERVER")
                    s = traceback.extract_tb(e.__traceback__).format()
                    s = [e for e in s if not "venv" in e][-1]
                    s = s.split(", ")
                    file = s[0].split(os.path.sep)[-1][:-1]
                    line = s[1].split(" ")[1]
                    additional = f" ({file}:{line})"
                except Exception as a:
                    trace = log_traceback("LOGGING")
                    additional = " THEN " + str(a)
                    pass

                self.got_response({"type":"error", "data":{"message":str(e) + additional, "trace": trace}})
        
    def download(self, type, url, token=None):
        import subprocess

        def gdownload(self, folder, url):
            import gdown
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
            import mega
            try:
                mega.Mega().login().download_url(url, folder)
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            except:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})

        def curldownload(self, folder, url, check=True):
            if check:
                r = subprocess.run(["curl", "-IL", url], capture_output=True)
                content_type = r.stdout.decode('utf-8').split("content-type: ")[-1].split(";",1)[0].split("\n",1)[0].strip()
                if not content_type in {"application/zip", "binary/octet-stream", "application/octet-stream", "multipart/form-data"}:
                    self.got_response({"type":"downloaded", "data":{"message": f"Unsupport type ({content_type}): " + url}})
                    return
            r = subprocess.run(["curl", "-OJL", url], cwd=folder)
            if r.returncode == 0:
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            else:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})

        def hfdownload(self, folder, url, token, check=True):
            header = f"Authorization: Bearer {token}"

            if check:
                r = subprocess.run(["curl", "-ILH", header, url], capture_output=True)
                content_type = r.stdout.decode('utf-8').split("content-type: ")[-1].split(";",1)[0].split("\n",1)[0].strip()
                if not content_type in {"application/zip", "binary/octet-stream", "application/octet-stream", "multipart/form-data"}:
                    self.got_response({"type":"downloaded", "data":{"message": f"Unsupport type ({content_type}): " + url}})
                    return
            r = subprocess.run(["curl", "-OJLH", header, url], cwd=folder)
            if r.returncode == 0:
                self.got_response({"type":"downloaded", "data":{"message": "Success: " + url}})
            else:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})

        folder = os.path.join(self.wrapper.storage.path, type)

        self.got_response({"type":"downloaded", "data":{"message": "Downloading: " + url + " to " + type}})

        check = True
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        if 'drive.google' in url:
            thread = threading.Thread(target=gdownload, args=([self, folder, url]))
            thread.start()
            return
        if 'mega.nz' in url:
            thread = threading.Thread(target=megadownload, args=([self, folder, url]))
            thread.start()
            return
        if 'civitai.com' in url:
            check = False
            if not "civitai.com/api/download/models" in url:
                self.got_response({"type":"downloaded", "data":{"message": "Failed: " + url}})
                return
        if 'huggingface' in url:
            url = url.replace("/blob/", "/resolve/")
            if token:
                thread = threading.Thread(target=hfdownload, args=([self, folder, url, token, check]))
                thread.start()
                return
        
        thread = threading.Thread(target=curldownload, args=([self, folder, url, check]))
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
    def __init__(self, wrapper, host, port, password=DEFAULT_PASSWORD, read_only=False, monitor=False):
        self.stopping = False

        self.requests = {}
        self.clients = {}

        self.scheme = None
        if password != None:
            self.scheme = get_scheme(password)

        self.monitor = monitor
        self.read_only = read_only
        self.owner = None

        self.inference = Inference(wrapper, read_only, callback=self.on_response)
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
        print("SERVER: shutdown")
        self.server.shutdown()
        print("SERVER: join")
        self.join()
        print("SERVER: done")

    def join(self, timeout=None):
        self.serve.join(timeout)
        if self.serve.is_alive():
            return False # timeout
        self.inference.join()
        return True

    def serve_forever(self):
        self.server.serve_forever()

    def handle_connection(self, connection):
        print(f"SERVER: client connected")
        client_id = get_id()

        self.clients[client_id] = queue.Queue()
        self.clients[client_id].put((-1, {"type":"hello", "data":{"id":client_id}}))

        if self.owner == None:
            self.owner = client_id
            self.inference.owner = self.owner
            self.clients[client_id].put((-1, {"type":"owner"}))

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
                                self.send_response(client_id, id, {'type': 'aborted', 'data': {}})

                        request_id = get_id()
                        user_id = request_id
                        if "id" in request:
                            user_id = request["id"]
                        self.requests[request_id] = client_id
                        mapping[request_id] = user_id

                        remaining = self.inference.requests.unfinished_tasks
                        self.inference.requests.put((client_id, request_id, request))
                        self.clients[client_id].put((-1, {"type":"ack", "data":{"id": user_id, "queue": remaining}}))
                    else:
                        self.clients[client_id].put((-1, {"type":"error", "data":{"message": error}}))
        except websockets.exceptions.WebSocketException:
            del self.clients[client_id]
        except Exception as e:
            del self.clients[client_id]
            log_traceback("CLIENT")
        print(f"SERVER: client disconnected")

    def send_response(self, client, id, response):
        if client in self.clients:
            self.clients[client].put((id, response))
        if client != self.owner and self.monitor and self.owner in self.clients:
            response = response.copy()
            response["monitor"] = True
            self.clients[self.owner].put((id, response))
        

    def on_response(self, id, response):
        if self.stopping:
            return False
        if id in self.requests:
            client = self.requests[id]
            self.send_response(client, id, response)
            return client in self.clients
        else:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sd-inference-server')
    parser.add_argument('--bind', type=str, help='address (ip:port) to listen on', default="127.0.0.1:28888")
    parser.add_argument('--password', type=str, help='password to derive encryption key from', default=DEFAULT_PASSWORD)
    parser.add_argument('--models', type=str, help='models folder', default="../../models")
    args = parser.parse_args()

    ip, port = args.bind.rsplit(":",1)

    model_storage = storage.ModelStorage(args.models, torch.float16, torch.float32)
    params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

    server = Server(params, ip, port, args.password)
    server.start()
    
    try:
        server.join()
    except KeyboardInterrupt:
        server.stop()
