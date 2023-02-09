import torch
import storage
import prompts
import utils
import attention

#pip install nvidia-pyindex
#pip install nvidia-tensorrt
#pip install torch-tensorrt==1.3.0 -f https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.3.0

device = torch.device("cuda")
dtype = torch.float16

storage = storage.ModelStorage("./models", torch.float16, torch.float32)

unet = storage.get_unet("SDv2", device)
clip = storage.get_clip("SDv2", device)

conditioning = prompts.ConditioningSchedule(clip, "confused cat looking up", "bad", 1, 2, 1)[0].to(dtype).to(device).detach()
noise = utils.NoiseSchedule([7], [], 512 // 8, 512 // 8, device, dtype)()
latents  = torch.cat([noise] * 2).detach()
timestep = torch.tensor(900).to(dtype).to(device).detach()

print(conditioning.shape)

torch.set_printoptions(threshold=10)

with torch.inference_mode():
    for i in range(10):
        with utils.CUDATimer():
            prediction = unet(latents, timestep, conditioning)
    
    if True:
        script = torch.jit.script(unet.eval())

        script_onnx = torch.onnx.export(script, (latents, timestep, conditioning), "onnx.pb")
        

    else:
        inputs = [
            torch_tensorrt.Input(
                min_shape=[2, 4, 32, 32],
                opt_shape=[2, 4, 64, 64],
                max_shape=[2, 4, 128, 128],
                dtype=torch.half,
            ),
            torch_tensorrt.Input(
                min_shape=[],
                opt_shape=[],
                max_shape=[],
                dtype=torch.half,
            ),
            torch_tensorrt.Input(
                min_shape=[2,77,768],
                opt_shape=[2,77,768],
                max_shape=[2,231,768],
                dtype=torch.half,
            )
        ]
        enabled_precisions = {torch.half}  # Run with fp16

        script = torch_tensorrt.compile(unet, inputs=inputs, enabled_precisions=enabled_precisions)
        #torch.jit.save(script, "trt_ts_module.ts")

    for i in range(10):
        with utils.CUDATimer():
            prediction2 = script(latents, timestep, conditioning)

    #torch.jit.save(script, "script.ts")

