import os
import sys
import glob
import PIL.Image
import torch
import numpy as np
import time
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "sd_scripts"))

from sd_scripts import train_network
from sd_scripts.library import train_util

def get_step_ckpt_name(args, ext: str, step_no: int):
    return f"{args.output_name}_{step_no}.{ext}"

train_network.train_util.get_step_ckpt_name = get_step_ckpt_name

class Dataset(train_util.MinimalDataset):
    def __init__(self, tokenizer, max_token_length, resolution, debug_dataset=False):
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)
        self.dataset = {}
        self.cache = {}
        self.batches = []

        self.bucket_manager = train_util.BucketManager(True, resolution, int(min(resolution)/2), int(max(resolution)*2), 64)
        self.bucket_manager.make_buckets()
        
        self.status_callback = None
        self.cache_callback = None
        self.batch_size = None

        self.last_time = time.time()

    def configure(self, batch_size, status_callback, cache_callback):
        self.batch_size = batch_size
        self.status_callback = status_callback
        self.cache_callback = cache_callback

    def add_pair(self, image, caption):
        name = f"MEMORY-{len(self.dataset)}"

        bucket, size, _ = self.bucket_manager.select_bucket(image.size[0], image.size[1])
        self.bucket_manager.add_image(bucket, name)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        self.dataset[name] = (image, caption, bucket, size)

        self.num_train_images = len(self.dataset)

    def add_path(self, path):
        for e in ["jpg", "jpeg", "png"]:
            for img in glob.glob(os.path.join(path, "*."+e)):
                image = PIL.Image.open(img)
                caption = ""
                for txt in [img + ".txt", img.rsplit(".",1)[0] + ".txt"]:
                    if os.path.exists(txt):
                        with open(txt, "r") as f:
                            caption = f.read()
                self.add_pair(image, caption)

    def __len__(self):
        if self.batches:
            return len(self.batches)
        return 1
    
    def is_latent_cacheable(self):
        return True
    
    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        self.cache = {}

        start = time.time()

        self.status_callback("Caching")

        for i, name in enumerate(tqdm.tqdm(self.dataset, "caching", smoothing=0)):
            if self.cache_callback:
                now = time.time()
                if now - self.last_time >= 2:
                    values = {
                        "n": i+1,
                        "total": len(self.dataset),
                        "elapsed": time.time() - start,
                    }
                    self.last = now
                    self.cache_callback(values)

            image, caption, bucket, size = self.dataset[name]

            image = np.array(image, np.uint8)
            image, _, _ = train_util.trim_and_resize_if_required(False, image, bucket, size)
            image = train_util.IMAGE_TRANSFORMS(image)

            img_tensors = torch.stack([image], dim=0)
            img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

            with torch.no_grad():
                latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

            self.cache[name] = latents
        
        self.get_batches()

        self.status_callback("Preparing")
    
    def get_batches(self):
        self.batches = []

        batch = []
        for reso, id in self.bucket_manager.reso_to_id.items():
            bucket = self.bucket_manager.buckets[id]    
            for i, name in enumerate(bucket):
                batch += [name]
                if len(batch) >= self.batch_size or i == len(bucket) - 1:
                    self.batches += [batch]
                    batch = []

    def __getitem__(self, idx):
        names = self.batches[idx]

        latents = torch.cat([self.cache[name] for name in names])
        captions = [self.dataset[name][1] for name in names]

        return {
            "latents": latents,
            "captions": captions,
            "loss_weights": 1
        }

class Trainer(train_network.NetworkTrainer):
    def __init__(self, status_callback, step_callback, cache_callback):
        super().__init__()
        self.args = None
        self.status_callback = status_callback
        self.step_callback = step_callback
        self.cache_callback = cache_callback
        self.dataset = None

        self.last_time = time.time()
        self.last_losses = []
        self.global_step = 0
        self.was_saving = False

    def assert_extra_args(self, args, train_dataset_group):
        self.dataset = train_dataset_group

        if self.folders:
            for folder in self.folders:
                self.dataset.add_path(folder)
        
        if self.pairs:
            for image, caption in self.pairs:
                self.dataset.add_pair(image, caption)

        self.dataset.configure(self.batch_size, self.status_callback, self.cache_callback)

    def log(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        if not "loss/current" in values:
            return
        
        if step == 1:
            self.status_callback("Training")
            self.start = time.time()

        self.global_step = step
        if self.args.save_every_n_steps:
            if self.args.max_train_steps - step < self.args.save_every_n_steps:
                self.args.save_every_n_steps = None
        
        if self.was_saving:
            self.status_callback("Training")
            self.was_saving = False

        self.last_losses += [values["loss/average"]]

        now = time.time()
        if now-self.last_time >= 2 or step == self.args.max_train_steps or step == 1:
            values = {
                "losses": self.last_losses.copy(),
                "n": step,
                "total": self.args.max_train_steps,
                "elapsed": time.time() - self.start,
                "epoch": len(self.dataset.batches)
            }
            self.last_losses = []
            self.last_time = now

            self.step_callback(values)

    def print(self, *args, **kwargs):
        s = args[0]
        if "saving checkpoint" in s:
            self.status_callback("Saving")
            self.was_saving = True
        
    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if not self.args:
            self.args = args
            self.accelerator = accelerator

            args.logging_dir = True
            accelerator.log = self.log
            accelerator.print = self.print

    def configure(self, params):
        lr_scheduler = params.learning_schedule.lower()
        if lr_scheduler == "cosine":
            lr_scheduler = "cosine_with_restarts"

        self.folders = params.folders
        self.pairs = params.dataset
        self.batch_size = params.batch_size

        if not self.folders and not self.pairs:
            raise Exception("No data")

        self.params = [
            f"--pretrained_model_name_or_path={params.base_model}",
            f"--clip_skip={params.clip_skip}",
            f"--output_dir={params.output_dir}",
            f"--output_name={params.name}",
            f"--lr_scheduler={lr_scheduler}",
            f"--lr_warmup_steps={int(params.warmup * params.steps)}",
            f"--lr_scheduler_num_cycles={int(params.restarts)}",
            f"--resolution={params.image_size}",
            f"--max_train_steps={params.steps}",
            f"--learning_rate={params.learning_rate}",
            f"--unet_lr={params.learning_rate}",
            f"--text_encoder_lr={params.learning_rate}",
            f"--network_dim={int(params.lora_rank)}",
            f"--network_alpha={int(params.lora_alpha)}",
            "--optimizer_type=AdamW",
            "--sdpa",
            "--mixed_precision=fp16",
            "--dataset_class=train.Dataset",
            "--cache_latents",
            "--network_module=networks.lora",
            "--enable_bucket",
            "--caption_extension=.txt",
            "--caption_separator=', '",
            "--shuffle_caption",
            "--weighted_captions",
            "--max_token_length=225",
            "--save_model_as=safetensors",
            "--save_every_n_steps=1000"
        ]

    def run(self):
        parser = train_network.setup_parser()
        args = parser.parse_args(self.params)
        args = train_network.train_util.read_config_from_file(args, parser)
        self.train(args)