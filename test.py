import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import storage
import lycoris.kohya

store = storage.ModelStorage("../../models", torch.float16, torch.float16)
store.find_all()

unet = store.get_unet("SD/azureanime_v5.safetensors", torch.device("cpu"))
vae  = store.get_vae("SD/azureanime_v5.safetensors", torch.device("cpu"))
clip = store.get_clip("SD/azureanime_v5.safetensors", torch.device("cpu"))

net, _ =  lycoris.kohya.create_network_from_weights(1.0, "v.safetensors", vae, clip.model, unet)


for l in net.unet_loras:
    print(type(l))
    break