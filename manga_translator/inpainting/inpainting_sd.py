import torch
import numpy as np
import cv2
import os
import gc
import einops
import safetensors
import safetensors.torch
from PIL import Image
from omegaconf import OmegaConf

from .common import OfflineInpainter
from ..utils import resize_keep_aspect

from .booru_tagger import Tagger
from .sd_hack import hack_everything
from .ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    return model


def load_ldm_sd(model, path) :
    if path.endswith('.safetensor') :
        sd = safetensors.torch.load_file(path)
    else :
        sd = load_state_dict(path)
    model.load_state_dict(sd, strict = False)

class StableDiffusionInpainter(OfflineInpainter):
    _MODEL_MAPPING = {
        'model_grapefruit': {
            'url': 'https://civitai.com/api/download/models/8364',
            'hash': 'dd680bd77d553e095faf58ff8c12584efe2a9b844e18bcc6ba2a366b85caceb8',
            'file': 'abyssorangemix2_Hard-inpainting.safetensors',
        },
        'model_wd_swinv2': {
            'url': 'https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx',
            'hash': '04ec04fdf7db74b4fed7f4b52f52e04dec4dbad9e4d88d2d178f334079a29fde',
            'file': 'wd_swinv2.onnx',
        },
        'model_wd_swinv2_csv': {
            'url': 'https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/raw/main/selected_tags.csv',
            'hash': '8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368',
            'file': 'selected_tags.csv',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        self.tagger = Tagger(self._get_file_path('wd_swinv2.onnx'))
        use_device = device
        if isinstance(use_device, str) and use_device.startswith('cuda') and (not torch.cuda.is_available()):
            use_device = 'cpu'

        self.model = create_model('manga_translator/inpainting/guided_ldm_inpaint9_v15.yaml').to(use_device)
        load_ldm_sd(self.model, self._get_file_path('abyssorangemix2_Hard-inpainting.safetensors'))
        hack_everything()
        self.model.eval()
        self.device = use_device
        self.model = self.model.to(use_device)

    async def _unload(self):
        del self.model

    @torch.no_grad()
    async def _infer(self, image: np.ndarray, mask: np.ndarray, config, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        img_original = np.copy(image)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        height, width, _ = image.shape

        def _run_once(img0: np.ndarray, mask0: np.ndarray, size: int) -> np.ndarray:
            img = img0
            m = mask0
            if max(img.shape[0: 2]) > size:
                img = resize_keep_aspect(img, size)
                m = resize_keep_aspect(m, size)
            pad_size = 64
            h, w, _ = img.shape
            new_h = (pad_size - (h % pad_size)) + h if h % pad_size != 0 else h
            new_w = (pad_size - (w % pad_size)) + w if w % pad_size != 0 else w
            if new_h != h or new_w != w:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            self.logger.info(f'Inpainting resolution: {new_w}x{new_h}')

            tags = self.tagger.label_cv2_bgr(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.logger.info(f'tags={list(tags.keys())}')

            blacklist = set()
            pos_prompt = ','.join([x for x in tags.keys() if x not in blacklist]).replace('_', ' ')
            pos_prompt = 'masterpiece,best quality,' + pos_prompt
            neg_prompt = 'worst quality, low quality, normal quality,text,text,text,text'

            if self.device.startswith('cuda'):
                with torch.autocast(enabled=True, device_type='cuda'):
                    out = self.model.img2img_inpaint(
                        image=Image.fromarray(img),
                        c_text=pos_prompt,
                        uc_text=neg_prompt,
                        mask=Image.fromarray(m),
                        device=self.device,
                    )
            else:
                out = self.model.img2img_inpaint(
                    image=Image.fromarray(img),
                    c_text=pos_prompt,
                    uc_text=neg_prompt,
                    mask=Image.fromarray(m),
                    device=self.device,
                )

            img_inpainted = (einops.rearrange(out, '1 c h w -> h w c').cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            if new_h != height or new_w != width:
                img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation=cv2.INTER_LINEAR)
            return img_inpainted

        retry_sizes = [int(inpainting_size), 1024, 768, 512]
        seen = set()
        retry_sizes = [s for s in retry_sizes if s > 0 and (s not in seen and not seen.add(s))]

        last_oom: Exception | None = None
        for size in retry_sizes:
            try:
                img_inpainted = _run_once(image, mask, size)
                ans = img_inpainted * mask_original + img_original * (1 - mask_original)
                return ans
            except Exception as e:
                oom = isinstance(e, (torch.cuda.OutOfMemoryError, getattr(torch, 'OutOfMemoryError', RuntimeError)))
                if (not oom) or (not self.device.startswith('cuda')):
                    raise

                last_oom = e
                try:
                    self.logger.warning(f'CUDA OOM during SD inpainting (size={size}). Retrying with smaller inpainting_size...')
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

        raise RuntimeError(
            'StableDiffusion inpainting failed due to CUDA out of memory. '
            'Try lowering inpainting_size (e.g. 1024/768/512), closing other GPU programs, '
            'or switch inpainter to lama_mpe/lama_large.'
        ) from last_oom
