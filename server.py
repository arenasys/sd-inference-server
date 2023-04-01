import threading
import traceback
import queue
import os
import torch

import simple_websocket_server as ws_server
import bson
import time

import attention
import storage
import wrapper

import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet, InvalidToken

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
                print(request["type"])
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
                    s = traceback.extract_tb(e.__traceback__).format()
                    s = [e for e in s if not "site-packages" in e][-1]
                    s = s.split(", ")
                    file = s[0].split(os.path.sep)[-1][:-1]
                    line = s[1].split(" ")[1]
                    additional = f" ({file}:{line})"
                except Exception:
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
            r = subprocess.run(["curl", "-s", "-O", "-J", "-L", url], cwd=folder)
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


class Server(ws_server.WebSocketServer):
    class Connection(ws_server.WebSocket):
        def connected(self):
            self.id = self.server.on_connected(self)

        def handle_close(self):
            self.server.on_disconnected(self.id)

        def handle(self):
            try:
                data = self.data
                if self.scheme:
                    data = self.scheme.decrypt(base64.urlsafe_b64encode(bytes(data)))
                request = bson.loads(data)
                assert type(request["type"]) == str
                assert not "data" in request or type(request["data"]) == dict
            except Exception as e:
                print(e)
                self.send_error("malformed request")
                return
            self.id = self.server.on_request(self.id, request)

        def send(self, response):
            try:
                data = bson.dumps(response)
                if self.scheme:
                    data = base64.urlsafe_b64decode(self.scheme.encrypt(data))
                self.send_message(data)
            except Exception:
                self.close()
                return

        def send_error(self, error):
            response = {"type": "error", "data": {"message":error}}
            self.send(response)

    def __init__(self, wrapper, host, port, password=None, allow_downloading=False):
        super().__init__(host, port, Server.Connection, select_interval=0.01)
        self.inference = Inference(wrapper, callback=self.on_response)
        self.serve = threading.Thread(target=self.serve_forever)

        self.responses = queue.Queue()

        self.id = 0
        self.clients = {}
        self.cancelled = set()

        self.stay_alive = True

        self.allow_downloading = allow_downloading

        self.scheme = None
        if password != None:
            self.scheme = get_scheme(password)

    def start(self):
        print("SERVER: starting")
        self.inference.start()
        self.serve.start()

    def stop(self):
        print("SERVER: stopping")
        self.clients = {}
        self.inference.stay_alive = False
        self.stay_alive = False
        self.join()

    def join(self):
        self.inference.join()
        self.serve.join()

    def serve_forever(self):
        while self.stay_alive:
            self.handle_request()
            while not self.responses.empty():
                id, response = self.responses.get()
                if id in self.clients:
                    self.clients[id].send(response)
        self.close()

    def on_connected(self, connection):
        self.id += 1
        self.clients[self.id] = connection
        connection.scheme = self.scheme
        return self.id

    def on_disconnected(self, id):
        del self.clients[id]
        print(f"SERVER: client disconnected")

    def on_reset(self, id):
        self.id += 1
        self.clients[self.id] = self.clients[id]
        del self.clients[id]
        return self.id

    def on_request(self, id, request):
        if request["type"] == "stop":
            return self.on_reset(id)
        if request["type"] == "cancel":
            id = self.on_reset(id)
            self.responses.put((id, {'type': 'aborted', 'data': {}}))
            return id

        self.inference.requests.put((id, request))
        return id

    def on_response(self, id, response):
        self.responses.put((id, response))
        return id in self.clients

if __name__ == "__main__":
    attention.use_optimized_attention()

    model_storage = storage.ModelStorage("../../models", torch.float16, torch.float32)
    params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

    server = Server(params, "127.0.0.1", "28888")
    server.start()
    
    try:
        server.join()
    except KeyboardInterrupt:
        server.stop()
