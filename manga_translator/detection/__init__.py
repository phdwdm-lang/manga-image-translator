import numpy as np
import torch
#from .ctd_utils.utils import dispatch as ctd_dispatch
from .default import DefaultDetector

# 缓存字典
DETECTORS = {}

_CTD_AVAILABLE: bool | None = None


def _is_ctd_available() -> bool:
    global _CTD_AVAILABLE
    if _CTD_AVAILABLE is None:
        try:
            from .ctd import ComicTextDetector  # noqa: F401
            _CTD_AVAILABLE = True
        except Exception:
            _CTD_AVAILABLE = False
    return _CTD_AVAILABLE


def resolve_detector_key(detector_key='default'):
    if hasattr(detector_key, 'value'):
        detector_key = detector_key.value
    if detector_key == 'ctd' and not _is_ctd_available():
        return 'default'
    return detector_key

def get_detector(detector_key='default'):
    requested_key = detector_key
    detector_key = resolve_detector_key(detector_key)
    # 如果缓存里没有，就创建一个新的
    if detector_key not in DETECTORS:
        if detector_key == 'default':
            DETECTORS[detector_key] = DefaultDetector()
        elif detector_key == 'ctd':
            from .ctd import ComicTextDetector
            DETECTORS[detector_key] = ComicTextDetector()
        elif detector_key == 'none':
            from .none import NoneDetector
            DETECTORS[detector_key] = NoneDetector()
        else:
            # 如果有其他模型，可以在这里扩展，暂时只支持默认
            DETECTORS[detector_key] = DefaultDetector()

    resolved = DETECTORS[detector_key]
    # Cache alias for requested key if it resolved to something else (e.g., ctd -> default)
    if requested_key != detector_key and requested_key not in DETECTORS:
        DETECTORS[requested_key] = resolved
    return resolved

async def dispatch(
    detector_key: str,
    image: np.ndarray,
    detect_size: int,
    text_threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    invert: bool,
    gamma_correct: bool,
    rotate: bool,
    auto_rotate: bool = False,
    device: str | None = None,
    verbose: bool = False,
):
    detector_key = resolve_detector_key(detector_key)
    # 1. 获取检测器
    detector = get_detector(detector_key)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 只对支持 load() 的 detector 做懒加载（NoneDetector 没有 load）
    if hasattr(detector, 'load'):
        if not hasattr(detector, 'model') or detector.model is None:
            print(f"[Detection] 正在加载检测模型 ({device})...")
            await detector.load(device)

    # 2. 执行检测
    regions = await detector.detect(
        image,
        detect_size,
        text_threshold,
        box_threshold,
        unclip_ratio,
        invert,
        gamma_correct,
        rotate,
        auto_rotate,
        verbose
    )

    return regions

async def prepare(task=None):
    # 占位函数，防止报错
    pass

def unload():
    # 占位函数，防止报错
    pass