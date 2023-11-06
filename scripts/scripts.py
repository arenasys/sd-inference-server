import time
import io
import queue
import threading
import websockets.sync.client
import websockets.exceptions
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
import bson
import PIL.Image

DEFAULT_PASSWORD = "qDiffusion"
FRAGMENT_SIZE = 524288

def get_scheme(password):
    password = password.encode("utf8")
    h = hashes.Hash(hashes.SHA256())
    h.update(password)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=h.finalize()[:16],
        iterations=480000,
    )
    return AESGCM(kdf.derive(password))

def encrypt(scheme, obj):
    data = bson.dumps(obj)
    if scheme:
        nonce = secrets.token_bytes(16)
        data = nonce + scheme.encrypt(nonce, data, b"")
    return data

def decrypt(scheme, data):
    if scheme:
        data = scheme.decrypt(data[:16], data[16:], b"")
    obj = bson.loads(data)
    return obj

def encode_image(img):
    bytesio = io.BytesIO()
    img.save(bytesio, format='PNG')
    return bytesio.getvalue()

def decode_image(bytes):
    return PIL.Image.open(io.BytesIO(bytes))

class Connection():
    def __init__(self, endpoint, password=None):
        self.stopping = False
        self.requests = queue.Queue()
        self.responses = queue.Queue()
        self.endpoint = endpoint
        self.client = None
        self.thread = threading.Thread(target=self.run, daemon=True)

        self.scheme = None
        if not password:
            password = "qDiffusion"
        self.password = password

    def connect(self):
        if self.client:
            return
        self.on_response({"type": "status", "data": {"message": "Connecting"}})
        while not self.client and not self.stopping:
            try:
                self.client = websockets.sync.client.connect(self.endpoint, open_timeout=2, max_size=None)
            except TimeoutError:
                pass
            except ConnectionRefusedError:
                self.on_response({"type": "remote_error", "data": {"message": "Connection refused"}})
                return
            except Exception as e:
                self.on_response({"type": "remote_error", "data": {"message": str(e)}})
                return
        if self.stopping:
            return
        if self.client:
            self.on_response({"type": "status", "data": {"message": "Connected"}})
            self.requests.put({"type":"options"})

    def run(self):
        self.scheme = get_scheme(self.password)
        self.connect()
        while self.client and not self.stopping:
            try:
                while True:
                    try:
                        data = self.client.recv(0)
                        response = decrypt(self.scheme, data)
                        self.on_response(response)
                    except TimeoutError:
                        break
                
                try:
                    request = self.requests.get(False)
                    data = encrypt(self.scheme, request)
                    data = [data[i:min(i+FRAGMENT_SIZE,len(data))] for i in range(0, len(data), FRAGMENT_SIZE)]

                    self.client.send(data)
                except queue.Empty:
                    time.sleep(5/1000)

            except websockets.exceptions.ConnectionClosedOK:
                self.on_response({"type": "remote_error", "data": {"message": "Connection closed"}})
                break
            except Exception as e:
                if type(e) == InvalidTag or type(e) == IndexError:
                    self.on_response({"type": "remote_error", "data": {"message": "Incorrect password"}})
                else:
                    self.on_response({"type": "remote_error", "data": {"message": str(e)}})
                break
            
        if self.client:
            self.client.close()
            self.client = None

    def stop(self):
        self.stopping = True

    def on_request(self, request):
        self.requests.put(request)

    def on_response(self, response):
        self.responses.put(response)

    def do_connect(self):
        self.thread.start()
        self.do_request(None, "hello")

    def do_disconnect(self):
        self.stopping = True
        while self.client:
            time.sleep(5/1000)

    def do_request(self, request, response_type):
        if request:
            if not "id" in request:
                request["id"] = secrets.randbelow(2147483647)
            self.on_request(request)
        while True:
            response = self.responses.get(block=True)
            if response["type"] == response_type:
                return response
            if response["type"] in {"error", "remote_error", "aborted", "done"}:
                raise Exception(response)
            