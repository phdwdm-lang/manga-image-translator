import os
import sys
import shutil
import subprocess
import cv2
import numpy as np
import uvicorn
import asyncio
import base64
import tempfile
import uuid
import time
import threading
import glob
import json
import zipfile
from typing import Optional, Any

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import hashlib
import requests
import py3langid as langid
import re

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_APPDATA = os.environ.get('LOCALAPPDATA')
if _LOCAL_APPDATA:
    DEFAULT_HF_HOME = os.path.join(_LOCAL_APPDATA, 'MangaTranslationStudio', 'hf_home')
else:
    DEFAULT_HF_HOME = os.path.join(_BASE_DIR, 'models', 'hf_home')
if not os.environ.get('HF_HOME'):
    os.environ['HF_HOME'] = DEFAULT_HF_HOME
if not os.environ.get('HF_HUB_CACHE'):
    os.environ['HF_HUB_CACHE'] = os.path.join(os.environ['HF_HOME'], 'hub')
if not os.environ.get('TRANSFORMERS_CACHE'):
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'transformers')

# 引入官方核心组件
from manga_translator.manga_translator import MangaTranslator
from manga_translator.config import Config, Ocr

MOCR_REPO_ID = 'kha-white/manga-ocr-base'

def _csv_env_list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name, '')
    if not isinstance(v, str) or not v.strip():
        return default
    items = [x.strip() for x in v.split(',') if isinstance(x, str) and x.strip()]
    return items or default

def _save_upload_to_temp(file: UploadFile, prefix: str) -> str:
    raw_prefix = str(prefix or "upload")
    safe_prefix = os.path.basename(raw_prefix).replace("/", "_").replace("\\", "_")
    if not safe_prefix:
        safe_prefix = "upload"

    filename = getattr(file, "filename", None) or ""
    ext = os.path.splitext(str(filename))[1].lower()
    if not ext or len(ext) > 10:
        ext = ".bin"

    path = os.path.join(tempfile.gettempdir(), f"{safe_prefix}_{uuid.uuid4().hex}{ext}")
    try:
        try:
            file.file.seek(0)
        except Exception:
            pass
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        raise
    return path

HF_ENDPOINTS = _csv_env_list(
    'MTS_HF_ENDPOINTS',
    [
        'https://hf-mirror.com',
        'https://huggingface.co',
    ],
)

if not os.environ.get('HF_ENDPOINT'):
    try:
        if HF_ENDPOINTS and isinstance(HF_ENDPOINTS[0], str) and HF_ENDPOINTS[0].strip():
            os.environ['HF_ENDPOINT'] = HF_ENDPOINTS[0].strip()
    except Exception:
        pass

MOCR_DOWNLOAD = {
    'state': 'idle',
    'error': '',
    'endpoint': '',
    'attempts': [],
    'started_at': 0.0,
    'finished_at': 0.0,
    'downloaded_bytes': 0,
    'speed_bps': 0.0,
    'log_path': '',
}

_MOCR_LOCK = threading.Lock()
_MOCR_CANCEL = threading.Event()
_MOCR_PROC: Optional[subprocess.Popen] = None

LAMA_LARGE_URLS = _csv_env_list(
    'MTS_LAMA_LARGE_URLS',
    [
        'https://hf-mirror.com/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
        'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
    ],
)

LAMA_LARGE_SHA256 = '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935'

LAMA_LARGE_DOWNLOAD: dict[str, Any] = {
    'state': 'idle',
    'error': '',
    'url': '',
    'started_at': 0.0,
    'finished_at': 0.0,
    'downloaded_bytes': 0,
    'total_bytes': 0,
    'speed_bps': 0.0,
}

_LAMA_LARGE_LOCK = threading.Lock()
_LAMA_LARGE_CANCEL = threading.Event()


def _mocr_repo_cache_dir() -> str:
    hub = os.environ.get('HF_HUB_CACHE') or os.path.join(os.environ.get('HF_HOME', DEFAULT_HF_HOME), 'hub')
    return os.path.join(hub, 'models--kha-white--manga-ocr-base')


def _dir_size_bytes(path: str) -> int:
    total = 0
    try:
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
    except Exception:
        return 0
    return total


def _mocr_is_downloaded() -> bool:
    base = _mocr_repo_cache_dir()
    if not os.path.isdir(base):
        return False
    snapshots_dir = os.path.join(base, 'snapshots', '*')

    # MangaOCR 运行时除了权重文件，还会读取 processor/tokenizer/config 等。
    # 如果这些文件没齐全，transformers 会尝试联网拉取（导致国内环境报 ConnectionReset/timeout）。
    required_any_model = [
        os.path.join(snapshots_dir, 'model.safetensors'),
        os.path.join(snapshots_dir, 'model.safetensors.index.json'),
        os.path.join(snapshots_dir, 'pytorch_model.bin'),
        os.path.join(snapshots_dir, 'pytorch_model.bin.index.json'),
    ]
    # 缺少 preprocessor_config 时，transformers/huggingface_hub 往往会尝试联网拉取该文件。
    required_preprocessor = [
        os.path.join(snapshots_dir, 'preprocessor_config.json'),
    ]
    required_config = [
        os.path.join(snapshots_dir, 'config.json'),
    ]
    required_any_tokenizer = [
        os.path.join(snapshots_dir, 'tokenizer.json'),
        os.path.join(snapshots_dir, 'tokenizer_config.json'),
        os.path.join(snapshots_dir, 'vocab.json'),
        os.path.join(snapshots_dir, 'spiece.model'),
    ]

    def _has_any(patterns: list[str]) -> bool:
        return any(bool(glob.glob(p)) for p in patterns)

    if not _has_any(required_any_model):
        return False
    if not _has_any(required_preprocessor):
        return False
    if not _has_any(required_config):
        return False
    if not _has_any(required_any_tokenizer):
        return False
    return True


def _mocr_start_download_if_needed() -> bool:
    global MOCR_DOWNLOAD
    if _mocr_is_downloaded():
        return False
    if MOCR_DOWNLOAD.get("state") == "downloading":
        return False
    _MOCR_CANCEL.clear()
    t = threading.Thread(target=_download_mocr_background, daemon=True)
    t.start()
    return True


def _download_mocr_background():
    global MOCR_DOWNLOAD
    global _MOCR_PROC
    MOCR_DOWNLOAD['state'] = 'downloading'
    MOCR_DOWNLOAD['error'] = ''
    MOCR_DOWNLOAD['endpoint'] = ''
    MOCR_DOWNLOAD['attempts'] = []
    MOCR_DOWNLOAD['started_at'] = time.time()
    MOCR_DOWNLOAD['finished_at'] = 0.0
    MOCR_DOWNLOAD['downloaded_bytes'] = 0
    MOCR_DOWNLOAD['speed_bps'] = 0.0
    MOCR_DOWNLOAD['log_path'] = ''

    try:
        last_err: Exception | None = None
        for endpoint in HF_ENDPOINTS:
            try:
                if _MOCR_CANCEL.is_set():
                    raise RuntimeError('canceled')
                MOCR_DOWNLOAD['endpoint'] = endpoint

                # 在子进程里执行下载，避免临时修改当前进程的 HF_HUB_OFFLINE/HF_ENDPOINT
                # 对正在进行的翻译请求造成干扰。
                env = dict(os.environ)
                env.pop('HF_HUB_OFFLINE', None)
                env.pop('TRANSFORMERS_OFFLINE', None)
                env['HF_ENDPOINT'] = endpoint

                code = (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download(repo_id='{MOCR_REPO_ID}', cache_dir=r'{os.environ.get('HF_HUB_CACHE')}', resume_download=True)"
                )
                log_path = os.path.join(tempfile.gettempdir(), f"mocr_snapshot_{uuid.uuid4().hex}.log")
                MOCR_DOWNLOAD['log_path'] = log_path
                try:
                    log_f = open(log_path, 'w', encoding='utf-8', errors='replace')
                except Exception:
                    log_f = None
                if log_f is not None:
                    print(f"[MOCR] snapshot_download log: {log_path}")

                p = subprocess.Popen(
                    [sys.executable, '-c', code],
                    stdout=log_f or subprocess.DEVNULL,
                    stderr=log_f or subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
                with _MOCR_LOCK:
                    _MOCR_PROC = p

                last_probe_ts = 0.0
                last_probe_bytes = 0
                last_log_ts = 0.0
                last_progress_ts = time.time()
                while True:
                    if _MOCR_CANCEL.is_set():
                        try:
                            p.terminate()
                            p.wait(timeout=5)
                        except Exception:
                            try:
                                p.kill()
                            except Exception:
                                pass
                            try:
                                p.wait(timeout=5)
                            except Exception:
                                pass
                        raise RuntimeError('canceled')
                    rc = p.poll()
                    if rc is not None:
                        break

                    now = time.time()
                    # Heartbeat every ~5s: scan cache dir size to prove download is progressing.
                    if now - last_log_ts >= 5.0:
                        base = _mocr_repo_cache_dir()
                        sz = _dir_size_bytes(base) if os.path.isdir(base) else 0
                        speed = 0.0
                        if last_probe_ts > 0.0 and now > last_probe_ts:
                            speed = (sz - last_probe_bytes) / max(0.001, now - last_probe_ts)
                        if sz != last_probe_bytes:
                            last_progress_ts = now
                        last_probe_ts = now
                        last_probe_bytes = sz
                        MOCR_DOWNLOAD['downloaded_bytes'] = int(sz)
                        MOCR_DOWNLOAD['speed_bps'] = float(max(0.0, speed))
                        elapsed = now - float(MOCR_DOWNLOAD.get('started_at') or now)
                        try:
                            mb = sz / 1024 / 1024
                            sp = speed / 1024 / 1024
                            print(f"[MOCR] downloading... {mb:.1f} MB, {sp:.2f} MB/s, elapsed={elapsed:.0f}s, endpoint={endpoint}")
                        except Exception:
                            pass
                        last_log_ts = now

                    # If cache dir size has not changed for a long time, treat it as stuck and try next endpoint.
                    if now - last_progress_ts >= 180.0:
                        try:
                            p.terminate()
                            p.wait(timeout=5)
                        except Exception:
                            try:
                                p.kill()
                            except Exception:
                                pass
                        raise RuntimeError('no_progress_timeout')
                    time.sleep(0.2)

                if log_f is not None:
                    try:
                        log_f.close()
                    except Exception:
                        pass

                if p.returncode != 0:
                    raise RuntimeError(f'snapshot_download_failed: {p.returncode}')

                if _mocr_is_downloaded():
                    MOCR_DOWNLOAD['state'] = 'done'
                    MOCR_DOWNLOAD['finished_at'] = time.time()
                    MOCR_DOWNLOAD['error'] = ''
                    return
                MOCR_DOWNLOAD['attempts'].append({'endpoint': endpoint, 'error': 'no_required_files_downloaded'})
            except Exception as e:
                last_err = e
                MOCR_DOWNLOAD['attempts'].append({'endpoint': endpoint, 'error': str(e)})
                continue

        if isinstance(last_err, Exception) and str(last_err) == 'canceled':
            MOCR_DOWNLOAD['state'] = 'idle'
            MOCR_DOWNLOAD['finished_at'] = time.time()
            MOCR_DOWNLOAD['error'] = 'canceled'
            return

        MOCR_DOWNLOAD['state'] = 'error'
        MOCR_DOWNLOAD['finished_at'] = time.time()
        if last_err is not None:
            MOCR_DOWNLOAD['error'] = str(last_err)
        elif MOCR_DOWNLOAD.get('attempts'):
            MOCR_DOWNLOAD['error'] = str(MOCR_DOWNLOAD['attempts'][-1].get('error', 'download_failed'))
        else:
            MOCR_DOWNLOAD['error'] = 'download_failed'
    except Exception as e:
        if str(e) == 'canceled':
            MOCR_DOWNLOAD['state'] = 'idle'
            MOCR_DOWNLOAD['finished_at'] = time.time()
            MOCR_DOWNLOAD['error'] = 'canceled'
        else:
            MOCR_DOWNLOAD['state'] = 'error'
            MOCR_DOWNLOAD['finished_at'] = time.time()
            MOCR_DOWNLOAD['error'] = str(e)
    finally:
        with _MOCR_LOCK:
            _MOCR_PROC = None


def _mocr_import_offline_zip(zip_path: str) -> None:
    cache_dir = os.environ.get('HF_HUB_CACHE') or os.path.join(os.environ.get('HF_HOME', DEFAULT_HF_HOME), 'hub')
    os.makedirs(cache_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        if not infos:
            raise RuntimeError('empty_zip')

        segment = 'models--kha-white--manga-ocr-base/'
        repo_dir = os.path.join(cache_dir, 'models--kha-white--manga-ocr-base')
        any_written = False

        safe_files: list[tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            name = info.filename
            if not isinstance(name, str) or not name or name.endswith('/'):
                continue
            norm = os.path.normpath(name).replace('\\', '/')
            if norm.startswith('../') or norm.startswith('..\\') or norm.startswith('..') or os.path.isabs(norm):
                raise RuntimeError('zip_path_traversal')
            safe_files.append((info, norm))

        if not safe_files:
            raise RuntimeError('empty_zip')

        # Case A: zip already contains HF hub cache layout: models--kha-white--manga-ocr-base/...
        for info, norm in safe_files:
            idx = norm.find(segment)
            if idx < 0:
                continue
            rel_path = norm[idx:]
            if not rel_path.startswith(segment):
                continue
            out_path = os.path.join(cache_dir, rel_path)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            with zf.open(info, 'r') as src, open(out_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            any_written = True

        if any_written:
            return

        # Case B: zip contains snapshots/<rev>/... (without the models--... prefix). Convert to HF cache layout.
        snapshots_revs: list[str] = []
        for info, norm in safe_files:
            idx = norm.find('snapshots/')
            if idx < 0:
                continue
            rest = norm[idx + len('snapshots/') :]
            parts = rest.split('/', 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            rev, subpath = parts[0], parts[1]
            out_path = os.path.join(repo_dir, 'snapshots', rev, subpath)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            with zf.open(info, 'r') as src, open(out_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            any_written = True
            snapshots_revs.append(rev)

        if any_written and snapshots_revs:
            # Ensure refs/main exists so hf_hub_download can resolve "main".
            rev = snapshots_revs[0]
            refs_dir = os.path.join(repo_dir, 'refs')
            os.makedirs(refs_dir, exist_ok=True)
            try:
                with open(os.path.join(refs_dir, 'main'), 'w', encoding='utf-8') as f:
                    f.write(str(rev))
            except Exception:
                pass
            return

        # Case C: zip contains raw model files (config.json/tokenizer/model/... possibly under one top-level folder).
        # Convert into HF cache layout: models--.../snapshots/<offline_rev>/...
        norms = [n for _, n in safe_files]
        rev = f"offline-{uuid.uuid4().hex}"

        parts_list = [n.split('/') for n in norms]
        can_strip_one = bool(parts_list) and all(len(p) > 1 for p in parts_list)
        if can_strip_one:
            first = parts_list[0][0]
            if all(p[0] == first for p in parts_list):
                stripped_norms = ['/'.join(p[1:]) for p in parts_list]
            else:
                stripped_norms = norms
        else:
            stripped_norms = norms

        # Some offline packs contain an extra repo folder like "manga-ocr-base/config.json".
        # Detect a plausible root dir from key files and strip it to flatten into snapshots/<rev>/.
        key_basenames = {
            'config.json',
            'preprocessor_config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json',
            'spiece.model',
            'model.safetensors',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'pytorch_model.bin.index.json',
        }
        root_dir: str | None = None
        for rel in stripped_norms:
            try:
                if os.path.basename(rel) == 'config.json':
                    root_dir = os.path.dirname(rel)
                    break
            except Exception:
                continue
        if root_dir is None:
            for rel in stripped_norms:
                try:
                    if os.path.basename(rel) in key_basenames:
                        root_dir = os.path.dirname(rel)
                        break
                except Exception:
                    continue

        for (info, _norm), rel in zip(safe_files, stripped_norms):
            rel = os.path.normpath(rel).replace('\\', '/')
            if not rel or rel.endswith('/'):
                continue
            if root_dir:
                try:
                    rd = os.path.normpath(root_dir).replace('\\', '/').strip('/')
                    if rd and rel.startswith(rd + '/'):
                        rel = rel[len(rd) + 1 :]
                except Exception:
                    pass
            out_path = os.path.join(repo_dir, 'snapshots', rev, rel)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            with zf.open(info, 'r') as src, open(out_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            any_written = True

        if not any_written:
            raise RuntimeError('invalid_zip_root')

        refs_dir = os.path.join(repo_dir, 'refs')
        os.makedirs(refs_dir, exist_ok=True)
        try:
            with open(os.path.join(refs_dir, 'main'), 'w', encoding='utf-8') as f:
                f.write(str(rev))
        except Exception:
            pass
        return


def _lama_large_file_path() -> str:
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter

    inp = LamaLargeInpainter()
    return inp._get_file_path('lama_large_512px.ckpt')


def _lama_large_is_downloaded() -> bool:
    return os.path.isfile(_lama_large_file_path())


def _lama_large_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest().lower()


def _download_lama_large_background():
    global LAMA_LARGE_DOWNLOAD
    with _LAMA_LARGE_LOCK:
        LAMA_LARGE_DOWNLOAD['state'] = 'downloading'
        LAMA_LARGE_DOWNLOAD['error'] = ''
        LAMA_LARGE_DOWNLOAD['url'] = ''
        LAMA_LARGE_DOWNLOAD['started_at'] = time.time()
        LAMA_LARGE_DOWNLOAD['finished_at'] = 0.0
        LAMA_LARGE_DOWNLOAD['downloaded_bytes'] = 0
        LAMA_LARGE_DOWNLOAD['total_bytes'] = 0
        LAMA_LARGE_DOWNLOAD['speed_bps'] = 0.0

    final_path = _lama_large_file_path()
    part_path = final_path + '.part'
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    try:
        last_err: Exception | None = None
        for url in LAMA_LARGE_URLS:
            try:
                if _LAMA_LARGE_CANCEL.is_set():
                    raise RuntimeError('canceled')

                existing = 0
                if os.path.isfile(part_path):
                    existing = os.path.getsize(part_path)

                headers: dict[str, str] = {'Accept-Encoding': 'identity'}
                if existing > 0:
                    headers['Range'] = f'bytes={existing}-'

                with _LAMA_LARGE_LOCK:
                    LAMA_LARGE_DOWNLOAD['url'] = url

                r = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30)
                if r.status_code == 416:
                    existing = 0
                    try:
                        os.remove(part_path)
                    except Exception:
                        pass
                    r = requests.get(url, stream=True, allow_redirects=True, headers={'Accept-Encoding': 'identity'}, timeout=30)

                if not r.ok:
                    raise RuntimeError(f'download_failed_http_{r.status_code}')

                total = int(r.headers.get('content-length', 0))
                total_bytes = total + existing if total else 0
                with _LAMA_LARGE_LOCK:
                    LAMA_LARGE_DOWNLOAD['total_bytes'] = total_bytes
                    LAMA_LARGE_DOWNLOAD['downloaded_bytes'] = existing

                start_ts = time.time()
                last_ts = start_ts
                last_bytes = existing
                wrote = existing

                with open(part_path, 'ab' if existing else 'wb') as f:
                    for data in r.iter_content(chunk_size=1024 * 128):
                        if _LAMA_LARGE_CANCEL.is_set():
                            raise RuntimeError('canceled')
                        if not data:
                            continue
                        f.write(data)
                        wrote += len(data)
                        now = time.time()
                        if now - last_ts >= 0.5:
                            speed = (wrote - last_bytes) / max(0.001, now - last_ts)
                            with _LAMA_LARGE_LOCK:
                                LAMA_LARGE_DOWNLOAD['downloaded_bytes'] = wrote
                                LAMA_LARGE_DOWNLOAD['speed_bps'] = float(speed)
                            last_ts = now
                            last_bytes = wrote

                with _LAMA_LARGE_LOCK:
                    LAMA_LARGE_DOWNLOAD['downloaded_bytes'] = wrote
                    LAMA_LARGE_DOWNLOAD['speed_bps'] = float((wrote - existing) / max(0.001, time.time() - start_ts))

                got = _lama_large_sha256(part_path)
                if got != LAMA_LARGE_SHA256.lower():
                    raise RuntimeError('sha256_mismatch')

                shutil.move(part_path, final_path)
                with _LAMA_LARGE_LOCK:
                    LAMA_LARGE_DOWNLOAD['state'] = 'done'
                    LAMA_LARGE_DOWNLOAD['error'] = ''
                    LAMA_LARGE_DOWNLOAD['finished_at'] = time.time()
                return
            except Exception as e:
                last_err = e
                continue

        with _LAMA_LARGE_LOCK:
            LAMA_LARGE_DOWNLOAD['state'] = 'error'
            LAMA_LARGE_DOWNLOAD['error'] = str(last_err) if last_err is not None else 'download_failed'
            LAMA_LARGE_DOWNLOAD['finished_at'] = time.time()
    except Exception as e:
        with _LAMA_LARGE_LOCK:
            LAMA_LARGE_DOWNLOAD['state'] = 'error'
            LAMA_LARGE_DOWNLOAD['error'] = str(e)
            LAMA_LARGE_DOWNLOAD['finished_at'] = time.time()


def _lama_large_import_offline_zip(zip_path: str) -> None:
    final_path = _lama_large_file_path()
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        if not infos:
            raise RuntimeError('empty_zip')

        target_names = {'lama_large_512px.ckpt'}
        picked: zipfile.ZipInfo | None = None
        for info in infos:
            name = info.filename
            if not isinstance(name, str) or not name or name.endswith('/'):
                continue
            norm = os.path.normpath(name).replace('\\', '/')
            if norm.startswith('../') or norm.startswith('..\\') or norm.startswith('..') or os.path.isabs(norm):
                raise RuntimeError('zip_path_traversal')
            base = os.path.basename(norm)
            if base in target_names:
                picked = info
                break

        if not picked:
            raise RuntimeError('missing_lama_large_ckpt')

        tmp_path = final_path + '.import.part'
        with zf.open(picked, 'r') as src, open(tmp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

        got = _lama_large_sha256(tmp_path)
        if got != LAMA_LARGE_SHA256.lower():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError('sha256_mismatch')

        shutil.move(tmp_path, final_path)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. 初始化 ---
def init_manga_translator():
    print("正在加载模型配置...")
    init_params = {
        'verbose': False, 'use_mtpe': False, 'font_path': None, 'models_ttl': 0,
        'batch_size': 1, 'ignore_errors': False, 'use_gpu': True,
        'use_gpu_limited': False, 'model_dir': 'models', 'kernel_size': 5,
        'input': [], 'save_text': False, 'load_text': False,
        'disable_memory_optimization': False, 'batch_concurrent': False,
        'context_size': 3
    }
    return MangaTranslator(init_params)

if os.environ.get('MTS_SKIP_MODEL_INIT') == '1':
    translator_instance = None
else:
    translator_instance = init_manga_translator()

MOCR_DISABLED = False
MOCR_DISABLED_REASON = ""

# 2. 补丁注入
import types
async def patched_run_detection(self, config, ctx):
    from manga_translator.detection import dispatch as dispatch_detection
    detector_key = config.detector.detector
    if hasattr(detector_key, 'value'):
        detector_key = detector_key.value
    return await dispatch_detection(
        detector_key,
        ctx.img_rgb,
        config.detector.detection_size,
        config.detector.text_threshold,
        config.detector.box_threshold,
        config.detector.unclip_ratio,
        config.detector.det_invert,
        config.detector.det_gamma_correct,
        config.detector.det_rotate,
        config.detector.det_auto_rotate,
        self.device,
        self.verbose,
    )
if translator_instance is not None:
    translator_instance._run_detection = types.MethodType(patched_run_detection, translator_instance)

async def patched_run_ocr(self, config, ctx):
    from manga_translator.ocr import dispatch as dispatch_ocr
    return await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device)
if translator_instance is not None:
    translator_instance._run_ocr = types.MethodType(patched_run_ocr, translator_instance)

print("系统准备就绪！")

RESULT_ROOT = os.path.join(_BASE_DIR, "result")
os.makedirs(RESULT_ROOT, exist_ok=True)

def _safe_task_name(raw: str) -> str:
    return os.path.basename(str(raw)).replace("/", "_").replace("\\", "_")

def _read_task_meta(folder_path: str) -> dict:
    meta_path = os.path.join(folder_path, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _write_task_meta(folder_path: str, meta: dict) -> None:
    meta_path = os.path.join(folder_path, "meta.json")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def _is_result_image_file(fn: str) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return False
    lower = fn.lower()
    if any(k in lower for k in ("-orig", "_orig", "orig-", "orig_", "source", "input", "mask", "inpaint", "det_", "ocr_", "debug")):
        return False
    return True


def _list_result_images_from_folder(folder_path: str, meta: dict | None = None) -> list[str]:
    files: list[str] = []
    if isinstance(meta, dict):
        raw = meta.get("files")
        if isinstance(raw, list):
            for x in raw:
                if not isinstance(x, str):
                    continue
                fn = os.path.basename(x).replace("/", "_").replace("\\", "_")
                if not fn:
                    continue
                p = os.path.join(folder_path, fn)
                if os.path.isfile(p) and _is_result_image_file(fn):
                    files.append(fn)
    if files:
        return files

    try:
        scanned = [
            fn
            for fn in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, fn)) and _is_result_image_file(fn)
        ]
        scanned.sort()
        return scanned
    except Exception:
        return []


async def _probe_lang_for_image(img_pil: Image.Image, detector_key: str = "ctd", detection_size: int = 1536) -> dict:
    try:
        from manga_translator.detection import resolve_detector_key

        img_rgb = np.array(img_pil.convert("RGB"))
        cfg = Config()
        cfg.detector.detector = resolve_detector_key(detector_key)
        cfg.detector.detection_size = detection_size
        cfg.detector.text_threshold = 0.3

        ctx = type("_TmpCtx", (), {})()
        ctx.img_rgb = img_rgb
        ctx.textlines = []

        ctx.textlines, _, _ = await translator_instance._run_detection(cfg, ctx)
        if not ctx.textlines:
            return {"detected_lang": "unknown", "confidence": 0.0, "text": ""}

        cfg.ocr.ocr = Ocr.ocr48px_ctc
        ctx.textlines = await translator_instance._run_ocr(cfg, ctx)
        texts = [getattr(tl, "text", "").strip() for tl in (ctx.textlines or [])]
        texts = [t for t in texts if t]
        merged = "\n".join(texts)
        if not merged.strip():
            return {"detected_lang": "unknown", "confidence": 0.0, "text": ""}

        detected_lang, confidence = langid.classify(merged)
        return {"detected_lang": detected_lang, "confidence": float(confidence), "text": merged[:300]}
    except Exception as e:
        return {"detected_lang": "unknown", "confidence": 0.0, "error": str(e)}


_TASK_ID_RE = re.compile(r".+_\d{8}_\d{6}$")


def _try_migrate_legacy_result_task(folder_name: str, folder_path: str, meta: dict) -> tuple[bool, dict]:
    if not isinstance(meta, dict):
        meta = {}

    legacy_title = meta.get("title") if isinstance(meta, dict) else None
    if isinstance(legacy_title, str) and legacy_title.strip():
        legacy_files = _list_result_images_from_folder(folder_path, meta)
        if not legacy_files:
            return False, meta
        try:
            meta["kind"] = "auto_translate"
            meta["files"] = legacy_files
            meta["updated_at"] = time.time()
            _write_task_meta(folder_path, meta)
        except Exception:
            pass
        return True, meta

    if not _TASK_ID_RE.match(folder_name or ""):
        return False, meta

    files = _list_result_images_from_folder(folder_path, meta)
    if not files:
        return False, meta

    base = folder_name
    try:
        parts = folder_name.rsplit("_", 2)
        if len(parts) == 3:
            base = parts[0]
    except Exception:
        base = folder_name

    try:
        meta["kind"] = "auto_translate"
        meta["title"] = meta.get("title") or base
        meta["files"] = files
        meta["updated_at"] = time.time()
        _write_task_meta(folder_path, meta)
    except Exception:
        pass
    return True, meta


@app.get("/results/list")
async def results_list(limit: int = 10):
    try:
        if not os.path.isdir(RESULT_ROOT):
            return {"directories": [], "items": []}

        folders: list[dict] = []
        for name in os.listdir(RESULT_ROOT):
            folder_path = os.path.join(RESULT_ROOT, name)
            if not os.path.isdir(folder_path):
                continue

            meta = _read_task_meta(folder_path)
            kind = meta.get("kind") if isinstance(meta, dict) else None
            if kind not in {"auto_translate", "manual_edit"}:
                if kind is None:
                    ok, meta = _try_migrate_legacy_result_task(name, folder_path, meta)
                    if not ok:
                        continue
                    kind = meta.get("kind")
                else:
                    continue

            files = _list_result_images_from_folder(folder_path, meta)
            if not files:
                continue

            try:
                mtime = os.path.getmtime(folder_path)
            except Exception:
                mtime = 0.0

            title = name
            try:
                t = meta.get("title")
                if isinstance(t, str) and t.strip():
                    title = t.strip()
            except Exception:
                title = name

            folders.append(
                {
                    "id": name,
                    "title": title,
                    "updated_at": mtime,
                    "cover": files[0],
                    "count": len(files),
                }
            )

        folders.sort(key=lambda x: x.get("updated_at", 0.0), reverse=True)
        if not isinstance(limit, int) or limit <= 0:
            limit = 10
        items = folders[: min(limit, len(folders))]
        return {
            "directories": [it["id"] for it in items],
            "items": items,
        }
    except Exception as e:
        return {"directories": [], "items": [], "error": str(e)}


@app.post("/results/upload_image")
async def results_upload_image(
    task: str = Form(...),
    filename: str = Form(...),
    display_title: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if not task or not filename:
        raise HTTPException(status_code=400, detail="Invalid task or filename")

    safe_task = _safe_task_name(task)
    safe_name = os.path.basename(str(filename)).replace("/", "_").replace("\\", "_")
    if not safe_task or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid task or filename")

    folder_path = os.path.join(RESULT_ROOT, safe_task)
    os.makedirs(folder_path, exist_ok=True)

    meta = _read_task_meta(folder_path)
    if not isinstance(meta, dict):
        meta = {}
    if meta.get("kind") not in {"auto_translate", "manual_edit"}:
        meta["kind"] = "auto_translate"
    if "created_at" not in meta:
        meta["created_at"] = time.time()

    if display_title is not None:
        try:
            title = str(display_title)
            meta["title"] = title
        except Exception:
            pass

    out_path = os.path.join(folder_path, safe_name)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(out_path, "wb") as f:
        f.write(data)

    try:
        files = meta.get("files")
        if not isinstance(files, list):
            files = []
        if safe_name not in files:
            files.append(safe_name)
        meta["files"] = files
        meta["updated_at"] = time.time()
        _write_task_meta(folder_path, meta)
    except Exception:
        pass

    return {
        "status": "ok",
        "task": safe_task,
        "filename": safe_name,
        "url": f"/results/file/{safe_task}/{safe_name}",
    }


@app.get("/results/file/{task}/{filename}")
async def results_file(task: str, filename: str):
    safe_task = _safe_task_name(task)
    safe_name = os.path.basename(str(filename)).replace("/", "_").replace("\\", "_")
    path = os.path.join(RESULT_ROOT, safe_task, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/results/pages/{task}")
async def results_pages(task: str):
    safe_task = _safe_task_name(task)
    folder_path = os.path.join(RESULT_ROOT, safe_task)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Task not found")

    meta = _read_task_meta(folder_path)
    kind = meta.get("kind") if isinstance(meta, dict) else None
    if kind not in {"auto_translate", "manual_edit"}:
        if kind is None:
            ok, meta = _try_migrate_legacy_result_task(safe_task, folder_path, meta)
            if not ok:
                raise HTTPException(status_code=404, detail="Task not found")
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    files = _list_result_images_from_folder(folder_path, meta)
    title = meta.get("title") if isinstance(meta, dict) else None
    if not isinstance(title, str):
        title = safe_task

    try:
        mtime = os.path.getmtime(folder_path)
    except Exception:
        mtime = 0.0

    return {
        "status": "ok",
        "task": safe_task,
        "title": title,
        "updated_at": mtime,
        "count": len(files),
        "files": files,
    }


@app.delete("/results/task/{task}")
async def results_delete_task(task: str):
    safe_task = _safe_task_name(task)
    folder_path = os.path.join(RESULT_ROOT, safe_task)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        shutil.rmtree(folder_path, ignore_errors=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete task failed: {e}")
    return {"status": "ok", "task": safe_task}


def _lama_large_start_download_if_needed() -> bool:
    if _lama_large_is_downloaded():
        return False
    with _LAMA_LARGE_LOCK:
        if LAMA_LARGE_DOWNLOAD.get('state') == 'downloading':
            return False
    _LAMA_LARGE_CANCEL.clear()
    t = threading.Thread(target=_download_lama_large_background, daemon=True)
    t.start()
    return True


def _extension_item_mocr() -> dict:
    cache_dir = _mocr_repo_cache_dir()
    installed = _mocr_is_downloaded()
    size_bytes = _dir_size_bytes(cache_dir) if os.path.isdir(cache_dir) else 0

    state = str(MOCR_DOWNLOAD.get('state') or ('done' if installed else 'idle'))
    if installed:
        state = 'done'
    return {
        'id': 'mocr',
        'name': 'MangaOCR (MOCR)',
        'description': '日文 OCR 模型（推荐用于日文漫画）',
        'size_bytes': int(size_bytes),
        'installed': bool(installed),
        'install_location': cache_dir,
        'download_state': state,
        'download_error': str(MOCR_DOWNLOAD.get('error') or ''),
        'download_endpoint': str(MOCR_DOWNLOAD.get('endpoint') or ''),
        'download_attempts': MOCR_DOWNLOAD.get('attempts') if isinstance(MOCR_DOWNLOAD.get('attempts'), list) else [],
        'download_url': '',
        'downloaded_bytes': int(MOCR_DOWNLOAD.get('downloaded_bytes') or size_bytes or 0),
        'total_bytes': int(MOCR_DOWNLOAD.get('total_bytes') or 0),
        'speed_bps': float(MOCR_DOWNLOAD.get('speed_bps') or 0.0),
        'restart_recommended': False,
        'restart_required': False,
        'restart_reason': '',
    }


def _extension_item_lama_large() -> dict:
    path = _lama_large_file_path()
    installed = _lama_large_is_downloaded()
    try:
        size_bytes = os.path.getsize(path) if installed else 0
    except Exception:
        size_bytes = 0

    with _LAMA_LARGE_LOCK:
        state = str(LAMA_LARGE_DOWNLOAD.get('state') or ('done' if installed else 'idle'))
        if installed:
            state = 'done'
        item = {
            'id': 'lama_large',
            'name': 'LaMa Large',
            'description': '更强的修复/去字模型（体积较大）',
            'size_bytes': int(size_bytes),
            'installed': bool(installed),
            'install_location': os.path.dirname(path),
            'download_state': state,
            'download_error': str(LAMA_LARGE_DOWNLOAD.get('error') or ''),
            'download_endpoint': '',
            'download_attempts': [],
            'download_url': str(LAMA_LARGE_DOWNLOAD.get('url') or (LAMA_LARGE_URLS[0] if LAMA_LARGE_URLS else '')),
            'downloaded_bytes': int(LAMA_LARGE_DOWNLOAD.get('downloaded_bytes') or (size_bytes if installed else 0)),
            'total_bytes': int(LAMA_LARGE_DOWNLOAD.get('total_bytes') or 0),
            'speed_bps': float(LAMA_LARGE_DOWNLOAD.get('speed_bps') or 0.0),
            'restart_recommended': False,
            'restart_required': False,
            'restart_reason': '',
        }
    return item


@app.get('/extensions/list')
async def extensions_list():
    items = [_extension_item_mocr(), _extension_item_lama_large()]
    return {'status': 'ok', 'items': items}


@app.post('/extensions/install')
async def extensions_install(id: str = Form(...)):
    ext_id = str(id or '').strip()
    if ext_id == 'mocr':
        started = _mocr_start_download_if_needed()
        return {'status': 'ok', 'id': 'mocr', 'download_started': bool(started)}
    if ext_id == 'lama_large':
        started = _lama_large_start_download_if_needed()
        return {'status': 'ok', 'id': 'lama_large', 'download_started': bool(started)}
    raise HTTPException(status_code=400, detail=f'Unknown extension id: {ext_id}')


@app.post('/extensions/import')
async def extensions_import(id: str = Form(...), file: UploadFile = File(...)):
    ext_id = str(id or '').strip()
    filename = str(file.filename or '').lower()
    if not filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail='仅支持 .zip 离线包')

    temp_path = _save_upload_to_temp(file, f'ext_import_{ext_id}')
    try:
        try:
            if ext_id == 'mocr':
                _mocr_import_offline_zip(temp_path)
                if not _mocr_is_downloaded():
                    raise RuntimeError('mocr_required_files_missing')
                return {'status': 'ok', 'id': 'mocr'}
            if ext_id == 'lama_large':
                _lama_large_import_offline_zip(temp_path)
                return {'status': 'ok', 'id': 'lama_large'}
            raise HTTPException(status_code=400, detail=f'Unknown extension id: {ext_id}')
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail='离线包不是有效的 zip 文件')
        except RuntimeError as e:
            code = str(e)
            if code == 'empty_zip':
                raise HTTPException(status_code=400, detail='离线包为空（zip 内无文件）')
            if code == 'zip_path_traversal':
                raise HTTPException(status_code=400, detail='离线包路径不安全（包含非法路径）')
            if code == 'invalid_zip_root':
                raise HTTPException(
                    status_code=400,
                    detail='离线包结构不符合预期：未找到 models--kha-white--manga-ocr-base 目录。请确认使用的是官方 MOCR 离线包，或重新下载后再导入。',
                )
            if code == 'mocr_required_files_missing':
                raise HTTPException(
                    status_code=400,
                    detail='离线包导入后仍缺少必要文件（例如 preprocessor_config.json）。请确认离线包完整且未被解压/二次打包。',
                )
            raise HTTPException(status_code=400, detail=f'离线包导入失败：{code}')
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@app.post('/extensions/uninstall')
async def extensions_uninstall(id: str = Form(...)):
    ext_id = str(id or '').strip()
    if ext_id == 'mocr':
        cache_dir = _mocr_repo_cache_dir()
        try:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        MOCR_DOWNLOAD['state'] = 'idle'
        MOCR_DOWNLOAD['error'] = ''
        MOCR_DOWNLOAD['endpoint'] = ''
        MOCR_DOWNLOAD['attempts'] = []
        MOCR_DOWNLOAD['started_at'] = 0.0
        MOCR_DOWNLOAD['finished_at'] = 0.0
        MOCR_DOWNLOAD['downloaded_bytes'] = 0
        MOCR_DOWNLOAD['speed_bps'] = 0.0
        MOCR_DOWNLOAD['log_path'] = ''
        return {'status': 'ok', 'id': 'mocr'}

    if ext_id == 'lama_large':
        path = _lama_large_file_path()
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        part_path = path + '.part'
        try:
            if os.path.isfile(part_path):
                os.remove(part_path)
        except Exception:
            pass

        with _LAMA_LARGE_LOCK:
            LAMA_LARGE_DOWNLOAD['state'] = 'idle'
            LAMA_LARGE_DOWNLOAD['error'] = ''
            LAMA_LARGE_DOWNLOAD['url'] = ''
            LAMA_LARGE_DOWNLOAD['started_at'] = 0.0
            LAMA_LARGE_DOWNLOAD['finished_at'] = 0.0
            LAMA_LARGE_DOWNLOAD['downloaded_bytes'] = 0
            LAMA_LARGE_DOWNLOAD['total_bytes'] = 0
            LAMA_LARGE_DOWNLOAD['speed_bps'] = 0.0
        return {'status': 'ok', 'id': 'lama_large'}

    raise HTTPException(status_code=400, detail=f'Unknown extension id: {ext_id}')


async def _apply_deepseek_request_overrides(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    changed = False

    if isinstance(api_key, str) and api_key.strip():
        os.environ['DEEPSEEK_API_KEY'] = api_key.strip()
        changed = True
        try:
            import openai  # type: ignore

            openai.api_key = api_key.strip()
        except Exception:
            pass

    if isinstance(api_base, str) and api_base.strip():
        os.environ['DEEPSEEK_API_BASE'] = api_base.strip()
        changed = True

    if isinstance(model, str) and model.strip():
        os.environ['DEEPSEEK_MODEL'] = model.strip()
        changed = True

    if not changed:
        return

    try:
        from manga_translator.config import Translator
        from manga_translator.translators import unload as unload_translator

        await unload_translator(Translator.deepseek)
    except Exception:
        pass


@app.post("/scan")
async def scan_image(
    file: UploadFile = File(...),
    lang: str = Form("ja"),
    inpainter: str = Form("none"),
    detector: str = Form("ctd"),
    detection_size: int = Form(1536),
    inpainting_size: int = Form(2048),
    translator: str = Form("deepseek"),
    target_lang: str = Form("CHS"),
    ocr: str = Form("auto"),
    deepseek_api_key: Optional[str] = Header(None, alias="x-deepseek-api-key"),
    deepseek_api_base: Optional[str] = Header(None, alias="x-deepseek-api-base"),
    deepseek_model: Optional[str] = Header(None, alias="x-deepseek-model"),
):
    global MOCR_DISABLED, MOCR_DISABLED_REASON
    temp_path = None
    try:
        allowed_inpainters = {"none", "original", "lama_mpe", "lama_large"}
        if inpainter not in allowed_inpainters:
            inpainter = "none"

        if inpainter == 'lama_large' and (not _lama_large_is_downloaded()):
            return {
                'status': 'error',
                'message': 'LaMa Large (lama_large) 模型未安装：请在设置-拓展中下载安装或导入离线包。',
            }

        allowed_detectors = {"default", "ctd", "paddle"}
        if detector not in allowed_detectors:
            return {"status": "error", "message": f"Invalid detector: {detector}"}

        allowed_detection_sizes = {1024, 1536, 2048, 2560}
        if detection_size not in allowed_detection_sizes:
            return {"status": "error", "message": f"Invalid detection_size: {detection_size}"}

        allowed_inpainting_sizes = {516, 1024, 2048, 2560}
        if inpainting_size not in allowed_inpainting_sizes:
            return {"status": "error", "message": f"Invalid inpainting_size: {inpainting_size}"}

        allowed_ocrs = {"auto", "32px", "48px", "48px_ctc", "mocr"}
        if ocr not in allowed_ocrs:
            return {"status": "error", "message": f"Invalid ocr: {ocr}"}

        allowed_translators = {
            "deepseek",
            "google",
            "youdao",
            "baidu",
            "deepl",
            "papago",
            "caiyun",
            "chatgpt",
            "chatgpt_2stage",
            "gemini",
            "gemini_2stage",
            "groq",
            "none",
            "original",
        }
        if translator not in allowed_translators:
            return {"status": "error", "message": f"Invalid translator: {translator}"}

        if target_lang:
            target_lang = str(target_lang).strip().upper()
            if len(target_lang) > 16:
                return {"status": "error", "message": "Invalid target_lang"}

        save_fn = globals().get("_save_upload_to_temp")
        if not callable(save_fn):
            def save_fn(file: UploadFile, prefix: str) -> str:
                raw_prefix = str(prefix or "upload")
                safe_prefix = os.path.basename(raw_prefix).replace("/", "_").replace("\\", "_")
                if not safe_prefix:
                    safe_prefix = "upload"
                filename = getattr(file, "filename", None) or ""
                ext = os.path.splitext(str(filename))[1].lower()
                if not ext or len(ext) > 10:
                    ext = ".bin"
                path = os.path.join(tempfile.gettempdir(), f"{safe_prefix}_{uuid.uuid4().hex}{ext}")
                try:
                    try:
                        file.file.seek(0)
                    except Exception:
                        pass
                    with open(path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                except Exception:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                    raise
                return path

        temp_path = save_fn(file, "manga_scan")
        img_pil = Image.open(temp_path).convert('RGB')
        
        from manga_translator.detection import resolve_detector_key
        cfg = Config()
        cfg.detector.detector = resolve_detector_key(detector)

        cfg.detector.detection_size = detection_size
        cfg.detector.text_threshold = 0.3

        cfg.inpainter.inpainting_size = inpainting_size

        detected_lang = lang
        if lang == "auto":
            probe_fn = globals().get("_probe_lang_for_image")
            if not callable(probe_fn):
                raise RuntimeError("_probe_lang_for_image is not defined")
            probe = await probe_fn(img_pil, detector_key=detector, detection_size=detection_size)
            detected_lang = probe.get("detected_lang", "unknown")

        ocr_requested = ocr
        forced_ocr = ocr != "auto"
        fallback_reason = ""

        # OCR 路由（源语言）
        used_ocr = Ocr.ocr48px_ctc
        if detected_lang == "ja":
            used_ocr = Ocr.mocr
        elif detected_lang == "en":
            used_ocr = Ocr.ocr48px
            cfg.detector.det_auto_rotate = False
        elif detected_lang in {"zh", "ko"}:
            used_ocr = Ocr.ocr48px_ctc
            cfg.detector.det_auto_rotate = False
        else:
            # unknown fallback
            used_ocr = Ocr.ocr48px_ctc

        if ocr != "auto":
            if ocr == "32px":
                used_ocr = Ocr.ocr32px
            elif ocr == "48px":
                used_ocr = Ocr.ocr48px
            elif ocr == "48px_ctc":
                used_ocr = Ocr.ocr48px_ctc
            elif ocr == "mocr":
                used_ocr = Ocr.mocr

            cfg.detector.det_auto_rotate = used_ocr == Ocr.mocr

        if (not forced_ocr) and used_ocr == Ocr.mocr and MOCR_DISABLED:
            used_ocr = Ocr.ocr48px_ctc
            cfg.detector.det_auto_rotate = False
            fallback_reason = MOCR_DISABLED_REASON or "mocr_failed_torch_security_cached"
            print(
                f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
            )

        if used_ocr == Ocr.mocr and (not _mocr_is_downloaded()):
            _mocr_start_download_if_needed()
            if forced_ocr:
                return {
                    "status": "error",
                    "message": "MangaOCR (mocr) 模型未安装或不完整：请在设置-拓展导入 MOCR 离线模型包，或先执行 /mocr/download 完成下载。",
                }
            used_ocr = Ocr.ocr48px_ctc
            cfg.detector.det_auto_rotate = False
            if not fallback_reason:
                fallback_reason = "mocr_not_downloaded"
            print(
                f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
            )

        if used_ocr == Ocr.mocr:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        else:
            if os.environ.get('HF_HUB_OFFLINE') == '1':
                try:
                    del os.environ['HF_HUB_OFFLINE']
                except Exception:
                    pass
            if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
                try:
                    del os.environ['TRANSFORMERS_OFFLINE']
                except Exception:
                    pass

        cfg.ocr.ocr = used_ocr
        print(
            f"OCR select: requested={ocr_requested} detected_lang={detected_lang} selected_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)}"
        )
        
        cfg.translator.translator = translator
        cfg.translator.target_lang = target_lang or 'CHS'
        
        # [开启嵌字功能]
        cfg.inpainter.inpainter = inpainter
        cfg.render.renderer = 'manga2Eng'      # 开启渲染器
        cfg.colorizer.colorizer = 'none'       # 暂时不到着色，保证文字清晰

        print(f"正在生成翻译画面 (inpainter={inpainter})...")
        try:
            if translator == 'deepseek':
                await _apply_deepseek_request_overrides(deepseek_api_key, deepseek_api_base, deepseek_model)
            ctx = await translator_instance.translate(img_pil, cfg)
        except Exception as e:
            msg = str(e)
            if (not forced_ocr) and used_ocr == Ocr.mocr and (
                'torch>=2.6' in msg
                or 'upgrade venv 内 torch' in msg
                or 'CVE-2025-32434' in msg
                or 'upgrade torch to at least v2.6' in msg
            ):
                used_ocr = Ocr.ocr48px_ctc
                cfg.ocr.ocr = used_ocr
                fallback_reason = "mocr_failed_torch_security"
                MOCR_DISABLED = True
                MOCR_DISABLED_REASON = fallback_reason
                cfg.detector.det_auto_rotate = False
                if os.environ.get('HF_HUB_OFFLINE') == '1':
                    try:
                        del os.environ['HF_HUB_OFFLINE']
                    except Exception:
                        pass
                if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
                    try:
                        del os.environ['TRANSFORMERS_OFFLINE']
                    except Exception:
                        pass
                print(
                    f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
                )
                ctx = await translator_instance.translate(img_pil, cfg)
            elif (not forced_ocr) and used_ocr == Ocr.mocr and (
                # 未导入离线包/本地 snapshot 不完整时，transformers/huggingface_hub 会尝试对 huggingface.co 发起 HEAD
                # 国内环境常见超时/连接重置，避免整次翻译失败：降级到 48px_ctc。
                'huggingface.co' in msg
                or 'HTTPSConnectionPool' in msg
                or 'MaxRetryError' in msg
                or 'ConnectTimeoutError' in msg
                or 'ConnectionResetError' in msg
                or 'thrown while requesting HEAD' in msg
                or '模型文件不存在或不完整' in msg
                or '无法离线加载' in msg
            ):
                used_ocr = Ocr.ocr48px_ctc
                cfg.ocr.ocr = used_ocr
                cfg.detector.det_auto_rotate = False
                fallback_reason = "mocr_offline_load_failed"
                if os.environ.get('HF_HUB_OFFLINE') == '1':
                    try:
                        del os.environ['HF_HUB_OFFLINE']
                    except Exception:
                        pass
                if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
                    try:
                        del os.environ['TRANSFORMERS_OFFLINE']
                    except Exception:
                        pass
                print(
                    f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason} err={msg}"
                )
                ctx = await translator_instance.translate(img_pil, cfg)
            elif used_ocr == Ocr.mocr and (
                # 未导入离线包/本地 snapshot 不完整时，transformers/huggingface_hub 会尝试对 huggingface.co 发起 HEAD
                # 国内环境常见超时/连接重置，避免整次翻译失败：降级到 48px_ctc。
                'huggingface.co' in msg
                or 'HTTPSConnectionPool' in msg
                or 'MaxRetryError' in msg
                or 'ConnectTimeoutError' in msg
                or 'ConnectionResetError' in msg
                or 'thrown while requesting HEAD' in msg
                or '模型文件不存在或不完整' in msg
                or '无法离线加载' in msg
            ):
                if forced_ocr:
                    return {
                        "status": "error",
                        "message": "MangaOCR (mocr) 加载失败：当前网络无法访问 HuggingFace 或本地离线模型不完整。\n请在 设置-拓展 导入 MOCR 离线包（确保包含 preprocessor_config.json），或配置代理/镜像（环境变量 HF_ENDPOINT 或 MTS_HF_ENDPOINTS）。",
                        "detail": msg[:600],
                    }

                used_ocr = Ocr.ocr48px_ctc
                cfg.ocr.ocr = used_ocr
                cfg.detector.det_auto_rotate = False
                fallback_reason = "mocr_offline_load_failed"
                if os.environ.get('HF_HUB_OFFLINE') == '1':
                    try:
                        del os.environ['HF_HUB_OFFLINE']
                    except Exception:
                        pass
                if os.environ.get('TRANSFORMERS_OFFLINE') == '1':
                    try:
                        del os.environ['TRANSFORMERS_OFFLINE']
                    except Exception:
                        pass
                print(
                    f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason} err={msg}"
                )
                ctx = await translator_instance.translate(img_pil, cfg)
            else:
                raise


        print(
            f"OCR final: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} fallback_reason={fallback_reason}"
        )
        # Base64 of inpainted "clean plate" (background without text rendering)
        clean_image_base64 = ""
        try:
            if getattr(ctx, 'img_inpainted', None) is not None:
                clean_arr = np.array(ctx.img_inpainted)
                if clean_arr.ndim == 2:
                    clean_arr = np.stack([clean_arr, clean_arr, clean_arr], axis=-1)
                if clean_arr.ndim == 3 and clean_arr.shape[-1] >= 4:
                    clean_arr = clean_arr[..., :3]

                if clean_arr.dtype != np.uint8:
                    try:
                        maxv = float(np.nanmax(clean_arr))
                    except Exception:
                        maxv = 255.0

                    if np.issubdtype(clean_arr.dtype, np.floating) and maxv <= 1.5:
                        clean_arr = clean_arr * 255.0

                    clean_arr = np.clip(clean_arr, 0, 255).astype(np.uint8)

                clean_buffer = BytesIO()
                clean_img = Image.fromarray(clean_arr).convert("RGB")
                clean_img.save(clean_buffer, format="JPEG", quality=90)
                clean_image_base64 = base64.b64encode(clean_buffer.getvalue()).decode()
        except Exception:
            import traceback
            traceback.print_exc()
            clean_image_base64 = ""

        # 将生成的“熟肉”图片转为 Base64
        translated_image_base64 = ""
        if ctx.result:
            buffered = BytesIO()
            # 确保是 RGB 模式再保存为 JPEG
            final_img = ctx.result.convert("RGB")
            final_img.save(buffered, format="JPEG", quality=90)
            translated_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        result_regions = []
        if ctx.text_regions:
            for region in ctx.text_regions:
                try:
                    if hasattr(region, 'xywh'): x, y, w, h = region.xywh
                    elif hasattr(region, 'box'): x, y, w, h = region.box
                    elif hasattr(region, 'pts'):
                        pts = np.array(region.pts).astype(int)
                        x, y, w, h = cv2.boundingRect(pts)
                    else: continue
                    
                    polygon = None
                    try:
                        if hasattr(region, 'min_rect'):
                            polygon = np.array(region.min_rect[0]).astype(int).tolist()
                        elif hasattr(region, 'pts'):
                            polygon = np.array(region.pts).astype(int).tolist()
                    except Exception:
                        polygon = None

                    fg_color = None
                    bg_color = None
                    try:
                        if hasattr(region, 'get_font_colors'):
                            fg, bg = region.get_font_colors()
                            fg_color = [int(fg[0]), int(fg[1]), int(fg[2])]
                            bg_color = [int(bg[0]), int(bg[1]), int(bg[2])]
                    except Exception:
                        fg_color = None
                        bg_color = None

                    result_regions.append({
                        "box": [int(x), int(y), int(w), int(h)],
                        "polygon": polygon,
                        "angle": float(getattr(region, 'angle', 0) or 0),
                        "text_original": region.text if hasattr(region, 'text') else "",
                        "text_translated": getattr(region, 'translation', ''),
                        "font_size": int(getattr(region, 'font_size', 0) or 0),
                        "direction": str(getattr(region, 'direction', 'auto') or 'auto'),
                        "alignment": str(getattr(region, 'alignment', 'auto') or 'auto'),
                        "letter_spacing": float(getattr(region, 'letter_spacing', 1.0) or 1.0),
                        "line_spacing": float(getattr(region, 'line_spacing', 1.0) or 1.0),
                        "fg_color": fg_color,
                        "bg_color": bg_color,
                    })
                except: continue


        return {
            "status": "success",
            "image_size": [img_pil.width, img_pil.height],
            "regions": result_regions,
            "detected_lang": detected_lang,
            "ocr_requested": ocr_requested,
            "ocr_used": used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr),
            "fallback_reason": fallback_reason,
            "used_ocr": used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr),
            "used_detector": cfg.detector.detector,
            "translated_image": f"data:image/jpeg;base64,{translated_image_base64}",
            "clean_image": f"data:image/jpeg;base64,{clean_image_base64}" if clean_image_base64 else ""
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

from pydantic import BaseModel

class TranslateTextRequest(BaseModel):
    text: str
    target_lang: str = "CHS"
    source_lang: str = "auto"

@app.post("/translate_text")
async def translate_text(
    request: TranslateTextRequest,
    deepseek_api_key: Optional[str] = Header(None, alias="x-deepseek-api-key"),
    deepseek_api_base: Optional[str] = Header(None, alias="x-deepseek-api-base"),
    deepseek_model: Optional[str] = Header(None, alias="x-deepseek-model"),
):
    """Translate a single text string using the configured translator."""
    try:
        text = request.text
        target_lang = request.target_lang
        source_lang = request.source_lang
        
        if not text or not text.strip():
            return {"status": "error", "message": "Text is empty"}
        
        try:
            from manga_translator.config import Translator
            from manga_translator.translators import get_translator

            await _apply_deepseek_request_overrides(deepseek_api_key, deepseek_api_base, deepseek_model)

            translator = get_translator(Translator.deepseek)

            from_lang = "auto" if (not source_lang or source_lang == "auto") else str(source_lang).strip().upper()
            to_lang = str(target_lang).strip().upper()

            translated = await translator.translate(from_lang, to_lang, [text.strip()])
            if translated and len(translated) > 0 and translated[0]:
                return {
                    "status": "success",
                    "translated_text": translated[0],
                    "source_text": text.strip(),
                    "target_lang": to_lang,
                }
            return {
                "status": "error",
                "message": "Translation returned empty result",
            }
        except Exception as trans_err:
            print(f"Translation error: {trans_err}")
            return {
                "status": "error",
                "message": f"Translation failed: {str(trans_err)}",
            }
        
        return {"status": "error", "message": "Translation service unavailable"}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

class RenderTextPreviewRequest(BaseModel):
    text: str
    width: int
    height: int
    font_size: int = 16
    font_family: str = "sans-serif"
    fill: str = "#000000"
    alignment: str = "center"
    line_height: float = 1.0
    letter_spacing: float = 0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    direction: str = "horizontal"
    stroke_color: str = "#000000"
    stroke_width: float = 0

@app.post("/render_text_preview")
async def render_text_preview(request: RenderTextPreviewRequest):
    """Render a single text region and return as PNG base64 for live preview."""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import re as re_module
        import math
        
        width = max(10, request.width)
        request_height = max(10, request.height)
        font_size = max(8, request.font_size)
        
        # Parse fill color
        fill_color = (0, 0, 0, 255)
        if request.fill.startswith("#"):
            hex_color = request.fill[1:]
            if len(hex_color) == 6:
                fill_color = (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), 255)
        elif request.fill.startswith("rgb"):
            match = re_module.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", request.fill)
            if match:
                fill_color = (int(match.group(1)), int(match.group(2)), int(match.group(3)), 255)
        
        # Parse stroke color
        stroke_color = (0, 0, 0, 255)
        if request.stroke_color.startswith("#"):
            hex_color = request.stroke_color[1:]
            if len(hex_color) == 6:
                stroke_color = (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), 255)
        
        # Load font
        win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
        fonts_dir = os.path.join(win_dir, "Fonts")
        font_file_map = {
            "Microsoft YaHei": "msyh.ttc", "微软雅黑": "msyh.ttc",
            "SimSun": "simsun.ttc", "宋体": "simsun.ttc",
            "SimHei": "simhei.ttf", "黑体": "simhei.ttf",
            "sans-serif": "msyh.ttc", "serif": "simsun.ttc", "monospace": "consola.ttf",
        }
        
        font_path = ""
        req_family = str(request.font_family or "").strip()
        if req_family and os.path.exists(req_family):
            font_path = req_family
        else:
            font_path = os.path.join(fonts_dir, font_file_map.get(req_family, "msyh.ttc"))
        font = None
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                pass
        if not font:
            font = ImageFont.load_default()
        
        stroke_w = int(max(1, font_size * 0.15)) if request.stroke_width == 0 else int(request.stroke_width)
        pad = max(2, int(min(width, request_height) * 0.06))
        max_w = max(1, width - pad * 2)
        
        actual_width = width
        actual_height = request_height
        pil_img = PILImage.new("RGBA", (actual_width, actual_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        content_min_x = math.inf
        content_min_y = math.inf
        content_max_x = -math.inf
        content_max_y = -math.inf

        def _update_content_bbox(x0: float, y0: float, x1: float, y1: float) -> None:
            nonlocal content_min_x, content_min_y, content_max_x, content_max_y
            if x1 <= x0 or y1 <= y0:
                return
            content_min_x = min(content_min_x, x0)
            content_min_y = min(content_min_y, y0)
            content_max_x = max(content_max_x, x1)
            content_max_y = max(content_max_y, y1)

        # Create temporary image for measuring text
        temp_img = PILImage.new("RGBA", (width, 100), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        def measure_bbox(s: str):
            try:
                return temp_draw.textbbox((0, 0), s, font=font, stroke_width=stroke_w)
            except Exception:
                return (0, 0, int(len(s) * font_size * 0.6), font_size)

        def measure_text(s: str) -> tuple[int, int]:
            bbox = measure_bbox(s)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

        def _draw_text_with_bold(x: int, y: int, s: str):
            bb = measure_bbox(s)
            dx = -min(0, bb[0])
            dy = -min(0, bb[1])
            if request.bold:
                for ox in [-1, 0, 1]:
                    for oy in [-1, 0, 1]:
                        if ox == 0 and oy == 0:
                            continue
                        draw.text((x + dx + ox, y + dy + oy), s, font=font, fill=fill_color, stroke_width=0)
            draw.text((x + dx, y + dy), s, font=font, fill=fill_color, stroke_width=stroke_w, stroke_fill=stroke_color)
        
        text = request.text or ""
        is_vertical = str(request.direction or "").strip().lower() in ("v", "vertical", "vr")
        
        # Calculate lines and actual height needed
        base_line_h = int(font_size * 1.2)
        line_h = int(base_line_h * request.line_height)

        # Frontend sends absolute pixel values (0 = no extra spacing, positive = add space, negative = reduce space)
        extra_spacing = int(round(request.letter_spacing)) if request.letter_spacing != 0 else 0
        
        if is_vertical:
            text_chars = list(text.replace("\n", ""))

            # Vertical mode semantics:
            # - letter_spacing: character spacing (vertical step)
            # - line_height: column spacing (horizontal step)
            base_char_h = int(font_size * 1.2)
            char_step = max(1, base_char_h + extra_spacing)
            max_h = max(1, actual_height - pad * 2)
            chars_per_col = max(1, max_h // char_step)
            num_cols = (len(text_chars) + chars_per_col - 1) // chars_per_col if text_chars else 1

            base_col_w = int(font_size * 1.2)
            col_gap = int(round(base_col_w * (request.line_height - 1.0)))
            col_step = max(1, base_col_w + col_gap)
            cols_total_w = base_col_w + max(0, num_cols - 1) * col_step

            bleed = max(12, stroke_w * 6, int(font_size * 0.9), abs(extra_spacing) * 4)
            need_w = pad * 2 + cols_total_w
            actual_width = max(width, need_w + bleed)
            if pil_img.size != (actual_width, actual_height):
                pil_img = PILImage.new("RGBA", (actual_width, actual_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(pil_img)

            max_w_v = max(1, actual_width - pad * 2)
            if request.alignment == "left":
                block_left = pad
            elif request.alignment == "center":
                block_left = pad + int((max_w_v - cols_total_w) / 2)
            else:
                block_left = pad + (max_w_v - cols_total_w)
            col0_x = block_left + max(0, num_cols - 1) * col_step
            start_y_v = pad

            char_idx = 0
            for col in range(num_cols):
                col_x = col0_x - col * col_step
                for row in range(chars_per_col):
                    if char_idx >= len(text_chars):
                        break
                    ch = text_chars[char_idx]
                    char_idx += 1

                    ch_y = start_y_v + row * char_step
                    ch_w, _ = measure_text(ch)
                    ch_x = col_x + (base_col_w - ch_w) // 2

                    bb_ch = measure_bbox(ch)
                    dx_ch = -min(0, bb_ch[0])
                    dy_ch = -min(0, bb_ch[1])
                    _update_content_bbox(
                        ch_x + dx_ch + bb_ch[0],
                        ch_y + dy_ch + bb_ch[1],
                        ch_x + dx_ch + bb_ch[2],
                        ch_y + dy_ch + bb_ch[3],
                    )
                    _draw_text_with_bold(ch_x, ch_y, ch)

            if request.italic:
                shear_factor = 0.2
                transform_matrix = (1, shear_factor, -shear_factor * actual_height, 0, 1, 0)
                try:
                    pil_img = pil_img.transform(
                        (actual_width, actual_height),
                        PILImage.AFFINE,
                        transform_matrix,
                        resample=PILImage.BICUBIC,
                    )
                except Exception:
                    pass

            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()

            content_bbox = [0, 0, 0, 0]
            if content_min_x != math.inf and content_min_y != math.inf and content_max_x != -math.inf and content_max_y != -math.inf:
                content_bbox = [
                    int(max(0, math.floor(content_min_x))),
                    int(max(0, math.floor(content_min_y))),
                    int(min(actual_width, math.ceil(content_max_x))),
                    int(min(actual_height, math.ceil(content_max_y))),
                ]

            return {
                "status": "success",
                "image": f"data:image/png;base64,{image_base64}",
                "width": actual_width,
                "height": actual_height,
                "content_bbox": content_bbox,
            }
        
        # Horizontal text layout (default)
        # Word wrap
        lines: list[str] = []
        for para in text.split("\n"):
            if not para:
                lines.append("")
                continue

            # IMPORTANT: line wrapping must match the actual draw logic.
            # When letter_spacing is non-zero, `measure_text(test)` underestimates the width,
            # causing right/left clipping. Here we accumulate per-character widths + spacing.
            line_chars: list[str] = []
            line_w = 0
            for ch in list(para):
                ch_w, _ = measure_text(ch)
                add_w = ch_w
                if line_chars:
                    add_w += extra_spacing
                if line_chars and (line_w + add_w) > max_w:
                    lines.append("".join(line_chars))
                    line_chars = [ch]
                    line_w = ch_w
                else:
                    line_chars.append(ch)
                    line_w += add_w
            if line_chars:
                lines.append("".join(line_chars))
        
        # Calculate line height with spacing
        base_line_h = int(font_size * 1.2)
        line_h = int(base_line_h * request.line_height)
        total_h = len(lines) * line_h

        # Calculate maximum line width (with letter spacing), used for horizontal overflow.
        max_line_w = 0
        for ln in lines:
            if not ln:
                continue
            w_ln = 0
            for j, ch in enumerate(ln):
                ch_w, _ = measure_text(ch)
                if j > 0:
                    w_ln += extra_spacing
                w_ln += ch_w
            max_line_w = max(max_line_w, w_ln)

        # If wrapped content exceeds the original textbox height, expand the preview image height.
        # Frontend renders the preview image without clipping, so larger images can overflow.
        max_h = max(1, actual_height - pad * 2)
        if total_h > max_h:
            bleed = max(12, stroke_w * 6, int(font_size * 0.9), abs(extra_spacing) * 4)
            need_h = pad * 2 + total_h
            actual_height = max(request_height, need_h + bleed)
            pil_img = PILImage.new("RGBA", (actual_width, actual_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pil_img)
            max_h = max(1, actual_height - pad * 2)

        # If any line is wider than the textbox, expand the preview image width to avoid left/right clipping.
        # Also add extra bleed for italic/stroke/spacing where glyph bounds can extend.
        bleed_x = max(12, stroke_w * 6, int(font_size * 0.9), abs(extra_spacing) * 4)
        if max_line_w > max_w or request.italic or extra_spacing != 0:
            need_w = pad * 2 + max_line_w
            actual_width = max(width, need_w + bleed_x)
            pil_img = PILImage.new("RGBA", (actual_width, actual_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pil_img)

        # Match vertical preview anchoring: top-aligned.
        start_y = pad

        fill = tuple(fill_color[:3])
        stroke = tuple(stroke_color[:3])
        max_w_h = max(1, actual_width - pad * 2)

        # Draw each line
        for i, ln in enumerate(lines):
            if not ln:
                continue

            # Calculate line width with letter spacing
            if extra_spacing != 0:
                total_ln_w = 0
                for ch in ln:
                    ch_w, _ = measure_text(ch)
                    total_ln_w += ch_w + extra_spacing
                total_ln_w -= extra_spacing
                ln_w = total_ln_w
            else:
                bb_ln = measure_bbox(ln)
                ln_w = bb_ln[2] - bb_ln[0]

            # Horizontal alignment (based on expanded canvas)
            xx = pad
            if request.alignment == "center":
                xx = pad + int((max_w_h - ln_w) / 2)
            elif request.alignment == "right":
                xx = pad + (max_w_h - ln_w)

            yy = start_y + i * line_h

            # Draw text with letter spacing
            if extra_spacing != 0:
                char_x = xx
                for ch in ln:
                    bb_ch = measure_bbox(ch)
                    dx_ch = -min(0, bb_ch[0])
                    dy_ch = -min(0, bb_ch[1])
                    draw_x = char_x + dx_ch
                    draw_y = yy + dy_ch
                    _update_content_bbox(draw_x + bb_ch[0], draw_y + bb_ch[1], draw_x + bb_ch[2], draw_y + bb_ch[3])
                    draw.text((draw_x, draw_y), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
                    ch_w = bb_ch[2] - bb_ch[0]
                    char_x += ch_w + extra_spacing
            else:
                bb_ln = measure_bbox(ln)
                dx_ln = -min(0, bb_ln[0])
                dy_ln = -min(0, bb_ln[1])
                draw_x = xx + dx_ln
                draw_y = yy + dy_ln
                _update_content_bbox(draw_x + bb_ln[0], draw_y + bb_ln[1], draw_x + bb_ln[2], draw_y + bb_ln[3])
                draw.text((draw_x, draw_y), ln, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)

            # Apply underline
            if request.underline:
                underline_y = yy + line_h - 2
                draw.line([(xx, underline_y), (xx + ln_w, underline_y)], fill=fill, width=max(1, int(font_size * 0.08)))

            # Apply strikethrough
            if request.strikethrough:
                strike_y = yy + line_h // 2
                draw.line([(xx, strike_y), (xx + ln_w, strike_y)], fill=fill, width=max(1, int(font_size * 0.08)))

        # Apply italic effect using affine transform on the rendered region
        if request.italic:
            shear_factor = 0.2
            try:
                w0, h0 = pil_img.size
                extra_w = int(h0 * shear_factor)
                new_w = int(w0 + extra_w)
                transform_matrix = (1, shear_factor, -shear_factor * h0, 0, 1, 0)
                pil_img = pil_img.transform(
                    (new_w, h0),
                    PILImage.AFFINE,
                    transform_matrix,
                    resample=PILImage.BICUBIC,
                )
                actual_width = new_w
                content_max_x = max(content_max_x, content_max_x + extra_w)
            except Exception:
                pass

        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        content_bbox = [0, 0, 0, 0]
        if content_min_x != math.inf and content_min_y != math.inf and content_max_x != -math.inf and content_max_y != -math.inf:
            content_bbox = [
                int(max(0, math.floor(content_min_x))),
                int(max(0, math.floor(content_min_y))),
                int(min(actual_width, math.ceil(content_max_x))),
                int(min(actual_height, math.ceil(content_max_y))),
            ]

        return {
            "status": "success",
            "image": f"data:image/png;base64,{image_base64}",
            "width": actual_width,
            "height": actual_height,
            "content_bbox": content_bbox,
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/supported_fonts")
async def supported_fonts():
    win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
    fonts_dir = os.path.join(win_dir, "Fonts")

    from PIL import ImageFont
    import hashlib

    font_file_map = {
        "sans-serif": "msyh.ttc",
        "serif": "simsun.ttc",
        "monospace": "consola.ttf",
        "Microsoft YaHei": "msyh.ttc",
        "SimSun": "simsun.ttc",
        "SimHei": "simhei.ttf",
    }

    label_map = {
        "sans-serif": "无衬线",
        "serif": "衬线",
        "monospace": "等宽",
        "microsoft yahei": "微软雅黑",
        "simsun": "宋体",
        "simhei": "黑体",
        "arial-unicode-regular.ttf": "Arial Unicode（全字符）",
        "notsansmonocjk-vf.ttf.ttc": "Noto 等宽 CJK（可变字体）",
        "notosansmonocjk-vf.ttf.ttc": "Noto 等宽 CJK（可变字体）",
        "msgothic.ttc": "MS Gothic（日文）",
        "msyh.ttc": "微软雅黑",
        "msyh.ttf": "微软雅黑",
        "simsun.ttc": "宋体",
        "simhei.ttf": "黑体",
        "consola.ttf": "Consolas（等宽）",
    }

    def _mask_sig(ft_font, ch: str) -> tuple[int, int, str] | None:
        try:
            m = ft_font.getmask(ch, mode="L")
            b = bytes(m)
            h = hashlib.md5(b).hexdigest()
            return (m.size[0], m.size[1], h)
        except Exception:
            return None

    def _supports_cjk_font_file(path: str) -> bool:
        try:
            f = ImageFont.truetype(path, 24)
        except Exception:
            return False

        missing = _mask_sig(f, "\uE000")
        if missing is None:
            return True

        samples = ["你", "好", "汉", "的", "一", "。"]
        for ch in samples:
            sig = _mask_sig(f, ch)
            if sig is None:
                continue
            if sig != missing:
                return True
        return False

    options: list[dict[str, str]] = []
    for k, fname in font_file_map.items():
        p = os.path.join(fonts_dir, fname)
        if os.path.exists(p):
            options.append({"value": k, "label": label_map.get(str(k).lower(), k)})

    repo_fonts_dir = os.path.join(_BASE_DIR, "fonts")
    try:
        if os.path.isdir(repo_fonts_dir):
            for fn in os.listdir(repo_fonts_dir):
                lower = fn.lower()
                if not (lower.endswith(".ttf") or lower.endswith(".ttc") or lower.endswith(".otf")):
                    continue
                fp = os.path.join(repo_fonts_dir, fn)
                if os.path.exists(fp):
                    if not _supports_cjk_font_file(fp):
                        continue
                    label = label_map.get(lower)
                    if not label:
                        try:
                            family, _style = ImageFont.truetype(fp, 24).getname()
                            label = label_map.get(str(family).lower(), family)
                        except Exception:
                            label = fn
                    options.append({"value": fp, "label": label})
    except Exception:
        pass

    dedup: dict[str, dict[str, str]] = {}
    for opt in options:
        dedup[opt["value"]] = opt

    out = list(dedup.values())
    out.sort(key=lambda x: x["label"].lower())
    return {"status": "success", "fonts": out}

@app.post("/render_page")
async def render_page(
    file: UploadFile = File(...),
    regions: str = Form("[]"),
    target_lang: str = Form("CHS"),
    font_path: str = Form(""),
    line_spacing: Optional[int] = Form(None),
    disable_font_border: int = Form(0),
):
    temp_path = None
    try:
        try:
            regions_data = json.loads(regions or "[]")
            if not isinstance(regions_data, list):
                raise ValueError("regions must be a JSON array")
        except Exception as e:
            return {"status": "error", "message": f"Invalid regions JSON: {e}"}

        temp_path = _save_upload_to_temp(file, "manga_render")
        img_pil = Image.open(temp_path).convert('RGB')
        img = np.array(img_pil)

        from manga_translator.utils import TextBlock
        from manga_translator.rendering import parse_font_paths, render as render_textblock, text_render

        selected_font_path = ""
        try:
            parsed = parse_font_paths(str(font_path or ""))
            if parsed:
                selected_font_path = parsed[0]
        except Exception:
            selected_font_path = ""

        if not selected_font_path:
            win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
            fonts_dir = os.path.join(win_dir, "Fonts")
            candidates = [
                os.path.join(fonts_dir, "msyh.ttc"),
                os.path.join(fonts_dir, "msyh.ttf"),
                os.path.join(fonts_dir, "simsun.ttc"),
                os.path.join(fonts_dir, "simhei.ttf"),
                os.path.join(fonts_dir, "arial.ttf"),
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    selected_font_path = p
                    break

        if selected_font_path:
            try:
                text_render.set_font(selected_font_path)
            except Exception:
                pass

        if selected_font_path:
            print(f"render_page: font={selected_font_path}")
        else:
            print("render_page: font=<default>")

        win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
        fonts_dir = os.path.join(win_dir, "Fonts")
        font_file_map = {
            "Microsoft YaHei": "msyh.ttc",
            "微软雅黑": "msyh.ttc",
            "SimSun": "simsun.ttc",
            "宋体": "simsun.ttc",
            "SimHei": "simhei.ttf",
            "黑体": "simhei.ttf",
            "sans-serif": "msyh.ttc",
            "serif": "simsun.ttc",
            "monospace": "consola.ttf",
        }

        def _as_bool(v: Any) -> bool:
            if v is True or v == 1:
                return True
            if v is False or v == 0 or v is None:
                return False
            s = str(v).strip().lower()
            return s in ("true", "1", "yes", "y")

        def _fallback_draw(
            img_arr: np.ndarray,
            bbox: tuple[int, int, int, int],
            text: str,
            fontp: str,
            fs: int,
            fg: list[int],
            bg: list[int],
            *,
            direction: str,
            alignment: str,
            letter_spacing: float,
            line_spacing_region: float,
            font_family: str,
            bold: bool,
            italic: bool,
            underline: bool,
            strikethrough: bool,
        ) -> np.ndarray:
            pil_img = Image.fromarray(img_arr).convert("RGB")
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(pil_img)

            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            box_w = max(1, (x1 - x0))
            box_h = max(1, (y1 - y0))
            pad = max(2, int(min(box_w, box_h) * 0.06))
            max_w = max(1, box_w - pad * 2)
            max_h = max(1, box_h - pad * 2)
            translation = str(text or "")
            font_size = int(fs)
            fg_color = fg
            bg_color = bg
            
            # Load font with style - Map frontend font family to Windows font files
            # Determine font file to use
            actual_font_path = fontp or selected_font_path
            if font_family and font_family in font_file_map:
                candidate = os.path.join(fonts_dir, font_file_map[font_family])
                if os.path.exists(candidate):
                    actual_font_path = candidate
            if actual_font_path:
                try:
                    text_render.set_font(actual_font_path)
                except Exception:
                    pass

            font = None
            if actual_font_path and os.path.exists(actual_font_path):
                try:
                    font = ImageFont.truetype(actual_font_path, font_size)
                except Exception:
                    font = None
            if not font:
                font = ImageFont.load_default()

            stroke_w = 0 if disable_font_border else max(1, int(font_size * 0.15))
            
            # Split text into lines based on width
            def measure_bbox(s: str):
                try:
                    return draw.textbbox((0, 0), s, font=font, stroke_width=stroke_w)
                except Exception:
                    return (0, 0, int(len(s) * font_size * 0.6), font_size)

            def measure_text(s: str) -> tuple[int, int]:
                bb = measure_bbox(s)
                return (bb[2] - bb[0], bb[3] - bb[1])
            
            # Check if vertical text layout
            align_v = str(alignment or "").strip().lower()
            if align_v not in {"left", "center", "right"}:
                align_v = "left"
            
            is_vertical = str(direction or "").strip().lower() in ("v", "vertical", "vr")
            
            if is_vertical:
                # Vertical text layout: each character on its own line, columns from right to left
                text_chars = list(str(translation or "").replace("\n", ""))
                base_char_h = int(font_size * 1.2)
                char_gap = int(round(letter_spacing)) if letter_spacing != 0 else 0
                char_step = max(1, base_char_h + char_gap)
                
                chars_per_col = max(1, max_h // char_step)
                
                # Calculate number of columns needed
                num_cols = (len(text_chars) + chars_per_col - 1) // chars_per_col if text_chars else 1
                base_col_w = int(font_size * 1.2)
                col_gap = int(round(base_col_w * (float(line_spacing_region) - 1.0)))
                col_step = max(1, base_col_w + col_gap)

                block_w = base_col_w + max(0, num_cols - 1) * col_step

                left_edge = x0 + pad
                if align_v == "right":
                    left_edge = x0 + pad + max(0, (max_w - block_w))
                elif align_v == "center":
                    left_edge = x0 + pad + max(0, int((max_w - block_w) / 2))
                # start_x is the rightmost column x (columns go right->left)
                start_x = left_edge + (num_cols - 1) * col_step
                start_y_v = y0 + pad
                
                fill = tuple(fg_color[:3])
                stroke = tuple(bg_color[:3])
                
                char_idx = 0
                for col in range(num_cols):
                    col_x = start_x - col * col_step
                    for row in range(chars_per_col):
                        if char_idx >= len(text_chars):
                            break
                        ch = text_chars[char_idx]
                        char_idx += 1
                        
                        ch_y = start_y_v + row * char_step
                        
                        # Center character horizontally in column
                        bb = measure_bbox(ch)
                        ch_w = bb[2] - bb[0]
                        ch_x = col_x + (base_col_w - ch_w) // 2

                        if bold:
                            for ox in [-1, 0, 1]:
                                for oy in [-1, 0, 1]:
                                    if ox == 0 and oy == 0:
                                        continue
                                    draw.text((ch_x - min(0, bb[0]) + ox, ch_y - min(0, bb[1]) + oy), ch, font=font, fill=fill, stroke_width=0)
                        draw.text((ch_x - min(0, bb[0]), ch_y - min(0, bb[1])), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
                
                # Apply italic if needed
                if italic:
                    region_box = (x0, y0, x1, y1)
                    region_img = pil_img.crop(region_box)
                    shear_factor = 0.2
                    width, height = region_img.size
                    from PIL import Image as PILImage
                    transform_matrix = (1, shear_factor, -shear_factor * height, 0, 1, 0)
                    try:
                        transformed = region_img.transform(
                            (width, height),
                            PILImage.AFFINE,
                            transform_matrix,
                            resample=PILImage.BICUBIC
                        )
                        pil_img.paste(transformed, (x0, y0))
                    except Exception:
                        pass  # If transform fails, skip italic effect

                img = np.array(pil_img)
                return img
            
            # Horizontal text layout (default)
            # Word wrap
            # Frontend sends absolute pixel values (0 = no extra spacing, positive = add space, negative = reduce space)
            extra_spacing = int(round(letter_spacing)) if letter_spacing != 0 else 0

            lines: list[str] = []
            for para in str(translation or "").split("\n"):
                if not para:
                    lines.append("")
                    continue

                # IMPORTANT: keep wrapping consistent with /render_text_preview
                line_chars: list[str] = []
                line_w = 0
                for ch in list(para):
                    ch_w, _ = measure_text(ch)
                    add_w = ch_w
                    if line_chars:
                        add_w += extra_spacing
                    if line_chars and (line_w + add_w) > max_w:
                        lines.append("".join(line_chars))
                        line_chars = [ch]
                        line_w = ch_w
                    else:
                        line_chars.append(ch)
                        line_w += add_w
                if line_chars:
                    lines.append("".join(line_chars))
            
            # Calculate line height with spacing
            base_line_h = int(font_size * 1.2)
            line_h = int(base_line_h * line_spacing_region)
            total_h = len(lines) * line_h
            
            # Match preview anchoring: top-aligned (frontend anchors content_bbox top to bbox top)
            start_y = y0 + pad
            
            fill = tuple(fg_color[:3])
            stroke = tuple(bg_color[:3])
            
            # Draw each line
            for i, ln in enumerate(lines):
                if not ln:
                    continue
                
                # Calculate line width with letter spacing
                if extra_spacing != 0:
                    # Measure each character and add spacing
                    total_ln_w = 0
                    for ch in ln:
                        ch_w, _ = measure_text(ch)
                        total_ln_w += ch_w + extra_spacing
                    total_ln_w -= extra_spacing  # Remove last extra spacing
                    ln_w = total_ln_w
                else:
                    bb_ln = measure_bbox(ln)
                    ln_w = bb_ln[2] - bb_ln[0]
                    ln_h = bb_ln[3] - bb_ln[1]
                
                ln_h = line_h  # Use calculated line height
                
                # Horizontal alignment
                xx = x0 + pad
                if alignment == "center":
                    xx = x0 + pad + int((max_w - ln_w) / 2)
                elif alignment == "right":
                    xx = x0 + pad + (max_w - ln_w)
                
                yy = start_y + i * line_h
                
                # Draw text with letter spacing
                if extra_spacing != 0:
                    # Draw character by character with spacing
                    char_x = xx
                    for ch in ln:
                        bb_ch = measure_bbox(ch)
                        if bold:
                            for offset_x in [-1, 0, 1]:
                                for offset_y in [-1, 0, 1]:
                                    if offset_x == 0 and offset_y == 0:
                                        continue
                                    draw.text((char_x - min(0, bb_ch[0]) + offset_x, yy - min(0, bb_ch[1]) + offset_y), ch, font=font, fill=fill, stroke_width=0)
                        draw.text((char_x - min(0, bb_ch[0]), yy - min(0, bb_ch[1])), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
                        ch_w = bb_ch[2] - bb_ch[0]
                        char_x += ch_w + extra_spacing
                else:
                    bb_ln = measure_bbox(ln)
                    if bold:
                        for offset_x in [-1, 0, 1]:
                            for offset_y in [-1, 0, 1]:
                                if offset_x == 0 and offset_y == 0:
                                    continue
                                draw.text((xx - min(0, bb_ln[0]) + offset_x, yy - min(0, bb_ln[1]) + offset_y), ln, font=font, fill=fill, stroke_width=0)
                    draw.text((xx - min(0, bb_ln[0]), yy - min(0, bb_ln[1])), ln, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
                
                # Apply underline
                if underline:
                    underline_y = yy + ln_h - 2
                    draw.line([(xx, underline_y), (xx + ln_w, underline_y)], fill=fill, width=max(1, int(font_size * 0.08)))
                
                # Apply strikethrough
                if strikethrough:
                    strike_y = yy + ln_h // 2
                    draw.line([(xx, strike_y), (xx + ln_w, strike_y)], fill=fill, width=max(1, int(font_size * 0.08)))
            
            # Apply italic effect using affine transform on the rendered region
            if italic:
                # Extract the region, apply shear transform, then paste back
                region_box = (x0, y0, x1, y1)
                region_img = pil_img.crop(region_box)
                
                # Create italic shear transform (skew by ~12 degrees)
                shear_factor = 0.2
                width, height = region_img.size
                new_width = int(width + height * shear_factor)
                
                # Apply affine transform for italic effect
                from PIL import Image as PILImage
                italic_img = PILImage.new("RGB", (new_width, height), (255, 255, 255))
                
                # Use affine transform: x' = x + shear * y
                transform_matrix = (1, shear_factor, -shear_factor * height, 0, 1, 0)
                try:
                    transformed = region_img.transform(
                        (new_width, height),
                        PILImage.AFFINE,
                        transform_matrix,
                        resample=PILImage.BICUBIC
                    )
                    # Crop back to original size and paste
                    cropped = transformed.crop((0, 0, width, height))
                    pil_img.paste(cropped, (x0, y0))
                except Exception:
                    pass  # If transform fails, skip italic effect
            
            img = np.array(pil_img)
            return img

        rendered_count = 0

        for item in regions_data:
            if not isinstance(item, dict):
                continue

            visible = item.get("visible", True)
            if visible is False:
                continue

            translation = item.get("text_translated")
            if translation is None:
                translation = item.get("translation", "")
            translation = str(translation or "").strip()
            if not translation:
                continue

            polygon = item.get("polygon")
            box = item.get("box")

            if isinstance(polygon, list):
                pts: list[list[float]] = []
                for pt in polygon:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            pts.append([float(pt[0]), float(pt[1])])
                        except Exception:
                            continue
                if len(pts) == 4:
                    polygon = pts
                elif len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    polygon = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
                else:
                    polygon = None

            if not polygon and isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                except Exception:
                    polygon = None

            if not polygon or not isinstance(polygon, list) or len(polygon) != 4:
                continue

            try:
                poly_i = np.array(polygon, dtype=np.int32).reshape(4, 2)
                poly_np = poly_i.astype(np.float32)
            except Exception:
                continue

            fg_color = item.get("fg_color")
            bg_color = item.get("bg_color")

            if not (isinstance(fg_color, list) and len(fg_color) >= 3):
                fg_color = [255, 255, 255]
            if bg_color is not None and not (isinstance(bg_color, list) and len(bg_color) >= 3):
                bg_color = None
            if bg_color is None:
                bg_color = [0, 0, 0]

            try:
                fg_color = [int(max(0, min(255, int(fg_color[0])))), int(max(0, min(255, int(fg_color[1])))), int(max(0, min(255, int(fg_color[2]))))]
            except Exception:
                fg_color = [255, 255, 255]
            try:
                bg_color = [int(max(0, min(255, int(bg_color[0])))), int(max(0, min(255, int(bg_color[1])))), int(max(0, min(255, int(bg_color[2]))))]
            except Exception:
                bg_color = [0, 0, 0]

            try:
                diff = abs(fg_color[0] - bg_color[0]) + abs(fg_color[1] - bg_color[1]) + abs(fg_color[2] - bg_color[2])
                bg_luma = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
                if diff < 60:
                    fg_color = [0, 0, 0] if bg_luma > 127 else [255, 255, 255]
            except Exception:
                pass

            font_size = item.get("font_size")
            try:
                font_size = int(font_size) if font_size is not None else -1
            except Exception:
                font_size = -1

            if font_size <= 0:
                try:
                    ys = poly_np[:, 1]
                    h_est = float(np.max(ys) - np.min(ys))
                    font_size = int(max(12, min(256, h_est * 0.6)))
                except Exception:
                    font_size = 24

            font_family = str(item.get("font_family") or "").strip()

            direction = str(item.get("direction", "auto") or "auto").strip().lower()
            if direction in ("vertical", "v"):
                direction = "v"
            elif direction in ("horizontal", "h"):
                direction = "h"

            alignment = str(item.get("alignment", item.get("align", "left")) or "left").strip().lower()
            if alignment == "auto" or alignment not in {"left", "center", "right"}:
                alignment = "left"

            letter_spacing = item.get("letter_spacing", 0.0)
            line_spacing_region = item.get("line_spacing", 1.0)
            try:
                letter_spacing = float(letter_spacing)
            except Exception:
                letter_spacing = 0.0
            try:
                line_spacing_region = float(line_spacing_region)
            except Exception:
                line_spacing_region = 1.0

            bold = _as_bool(item.get("bold"))
            italic = _as_bool(item.get("italic"))
            underline = _as_bool(item.get("underline"))
            strikethrough = _as_bool(item.get("strikethrough"))

            # Keep layout width/height consistent with editor preview (use the original box).

            # Per-region font selection
            actual_font_path = selected_font_path
            if font_family:
                if os.path.exists(font_family):
                    actual_font_path = font_family
                else:
                    mapped = font_file_map.get(font_family)
                    if mapped:
                        candidate = os.path.join(fonts_dir, mapped)
                        if os.path.exists(candidate):
                            actual_font_path = candidate
            if actual_font_path:
                try:
                    text_render.set_font(actual_font_path)
                except Exception:
                    pass

            xs_i = poly_i[:, 0]
            ys_i = poly_i[:, 1]
            x0 = int(max(0, min(xs_i)))
            x1 = int(min(img.shape[1], max(xs_i)))
            y0 = int(max(0, min(ys_i)))
            y1 = int(min(img.shape[0], max(ys_i)))
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                continue

            img = _fallback_draw(
                img,
                (x0, y0, x1, y1),
                translation,
                actual_font_path,
                font_size,
                fg_color,
                bg_color,
                direction=direction,
                alignment=alignment,
                letter_spacing=letter_spacing,
                line_spacing_region=line_spacing_region,
                font_family=font_family,
                bold=bold,
                italic=italic,
                underline=underline,
                strikethrough=strikethrough,
            )

            rendered_count += 1

        buffered = BytesIO()
        final_img = Image.fromarray(img).convert("RGB")
        final_img.save(buffered, format="JPEG", quality=92)
        translated_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        print(f"render_page: total_regions={len(regions_data)} rendered_count={rendered_count}")

        return {
            "status": "success",
            "image_size": [final_img.width, final_img.height],
            "image": f"data:image/jpeg;base64,{translated_image_base64}",
            "rendered_count": rendered_count,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)