import threading
import traceback
import queue
import os
import torch
import ssl

import simple_websocket_server as ws_server
import bson

import attention
import storage
import wrapper

class Inference(threading.Thread):
    def __init__(self, wrapper, callback):
        super().__init__()
        
        self.wrapper = wrapper
        wrapper.callback = self.got_response

        self.callback = callback
        self.requests = queue.Queue()
        self.current = None

        self.stay_alive = True

    def got_response(self, response):
        return self.callback(self.current, response)

    def run(self):
        while self.stay_alive:
            try:
                self.current, request = self.requests.get(True, 0.1)
                self.wrapper.reset()
                if request["type"] == "txt2img":
                    self.wrapper.set(**request["data"])
                    self.wrapper.txt2img()
                elif request["type"] == "img2img":
                    self.wrapper.set(**request["data"])
                    self.wrapper.img2img()
                elif request["type"] == "options":
                    self.wrapper.options()
                elif request["type"] == "convert":
                    self.wrapper.set(**request["data"])
                    self.wrapper.convert()
                self.requests.task_done()
            except queue.Empty:
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


class Server(ws_server.WebSocketServer):
    class Connection(ws_server.WebSocket):
        def connected(self):
            self.id = self.server.on_connected(self)

        def handle_close(self):
            self.server.on_disconnected(self.id)

        def handle(self):
            try:
                request = bson.loads(self.data)
                assert type(request["type"]) == str
                assert not "data" in request or type(request["data"]) == dict
            except Exception as e:
                self.send_error("malformed request")
                return
            self.id = self.server.on_request(self.id, request)

        def send(self, response):
            try:
                response = bson.dumps(response)
                self.send_message(response)
            except Exception:
                self.close()
                return

        def send_error(self, error):
            response = {"type": "error", "data": {"message":error}}
            self.send(response)

    def __init__(self, wrapper, host, port, certfile=None, keyfile=None, ssl_version=ssl.PROTOCOL_TLSv1_2):
        super().__init__(host, port, Server.Connection, select_interval=0.01, certfile=certfile, keyfile=keyfile, ssl_version=ssl_version)
        self.inference = Inference(wrapper, callback=self.on_response)
        self.serve = threading.Thread(target=self.serve_forever)

        self.responses = queue.Queue()

        self.id = 0
        self.clients = {}
        self.cancelled = set()

        self.stay_alive = True

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
        print(f"SERVER: client connected")
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

    model_storage = storage.ModelStorage("../models", torch.float16, torch.float32)
    params = wrapper.GenerationParameters(model_storage, torch.device("cuda"))

    server = Server(params, "127.0.0.1", "28888")
    server.start()
    
    try:
        server.join()
    except KeyboardInterrupt:
        server.stop()
