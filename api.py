import os
import shutil
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
from typing import Optional, Any

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

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
HF_ENDPOINTS = [
    'https://hf-mirror.com',
    'https://huggingface.co',
]

MOCR_DOWNLOAD = {
    'state': 'idle',
    'error': '',
    'endpoint': '',
    'attempts': [],
    'started_at': 0.0,
    'finished_at': 0.0,
}


def _mocr_repo_cache_dir() -> str:
    hub = os.environ.get('HF_HUB_CACHE') or os.path.join(os.environ.get('HF_HOME', DEFAULT_HF_HOME), 'hub')
    return os.path.join(hub, 'models--kha-white--manga-ocr-base')


def _mocr_is_downloaded() -> bool:
    base = _mocr_repo_cache_dir()
    if not os.path.isdir(base):
        return False
    snapshots = os.path.join(base, 'snapshots', '*', 'model.safetensors')
    return len(glob.glob(snapshots)) > 0


def _download_mocr_background():
    global MOCR_DOWNLOAD
    try:
        os.environ.pop('HF_HUB_OFFLINE', None)
    except Exception:
        pass
    MOCR_DOWNLOAD['state'] = 'downloading'
    MOCR_DOWNLOAD['error'] = ''
    MOCR_DOWNLOAD['endpoint'] = ''
    MOCR_DOWNLOAD['attempts'] = []
    MOCR_DOWNLOAD['started_at'] = time.time()
    MOCR_DOWNLOAD['finished_at'] = 0.0

    try:
        from huggingface_hub import snapshot_download

        last_err = None
        for endpoint in HF_ENDPOINTS:
            try:
                os.environ['HF_ENDPOINT'] = endpoint
                MOCR_DOWNLOAD['endpoint'] = endpoint
                snapshot_download(
                    repo_id=MOCR_REPO_ID,
                    cache_dir=os.environ.get('HF_HUB_CACHE'),
                    resume_download=True,
                )
                if _mocr_is_downloaded():
                    MOCR_DOWNLOAD['state'] = 'done'
                    MOCR_DOWNLOAD['finished_at'] = time.time()
                    MOCR_DOWNLOAD['error'] = ''
                    return
                MOCR_DOWNLOAD['attempts'].append({'endpoint': endpoint, 'error': 'no_files_downloaded'})
            except Exception as e:
                last_err = e
                MOCR_DOWNLOAD['attempts'].append({'endpoint': endpoint, 'error': str(e)})
                continue

        MOCR_DOWNLOAD['state'] = 'error'
        MOCR_DOWNLOAD['finished_at'] = time.time()
        if last_err is not None:
            MOCR_DOWNLOAD['error'] = str(last_err)
        elif MOCR_DOWNLOAD.get('attempts'):
            MOCR_DOWNLOAD['error'] = str(MOCR_DOWNLOAD['attempts'][-1].get('error', 'download_failed'))
        else:
            MOCR_DOWNLOAD['error'] = 'download_failed'
    except Exception as e:
        MOCR_DOWNLOAD['state'] = 'error'
        MOCR_DOWNLOAD['finished_at'] = time.time()
        MOCR_DOWNLOAD['error'] = str(e)


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
translator_instance._run_detection = types.MethodType(patched_run_detection, translator_instance)

async def patched_run_ocr(self, config, ctx):
    from manga_translator.ocr import dispatch as dispatch_ocr
    return await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device)
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

def _save_upload_to_temp(upload: UploadFile, prefix: str) -> str:
    original_name = upload.filename or "upload"
    base = os.path.basename(original_name)
    # strip path separators that may survive basename on some inputs
    base = base.replace("/", "_").replace("\\", "_")
    ext = os.path.splitext(base)[1]
    temp_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return temp_path

def _detect_lang_from_text(text: str) -> str:
    if not text:
        return "unknown"

    counts = {
        "hiragana": 0,
        "katakana": 0,
        "han": 0,
        "hangul": 0,
        "latin": 0,
    }
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= o <= 0x30FF:
            counts["katakana"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["han"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            counts["hangul"] += 1
        elif (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A):
            counts["latin"] += 1

    if counts["hiragana"] + counts["katakana"] >= max(3, counts["latin"], counts["hangul"]):
        return "ja"
    if counts["hangul"] >= max(3, counts["latin"], counts["hiragana"] + counts["katakana"]):
        return "ko"
    if counts["latin"] >= max(3, counts["hangul"], counts["hiragana"] + counts["katakana"]):
        return "en"
    if counts["han"] >= 3:
        return "zh"
    return "unknown"


async def _probe_lang_for_image(img_pil: Image.Image, detector_key: str = "ctd", detection_size: int = 1536) -> dict:
    from manga_translator.detection import dispatch as dispatch_detection, resolve_detector_key
    from manga_translator.ocr import dispatch as dispatch_ocr

    img_rgb = np.array(img_pil.convert("RGB"))

    device = translator_instance.device
    if not device:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    detector_key = resolve_detector_key(detector_key)

    cfg = Config()
    cfg.detector.detector = detector_key
    cfg.detector.detection_size = detection_size
    cfg.detector.text_threshold = 0.3
    cfg.detector.det_auto_rotate = True

    textlines, _, _ = await dispatch_detection(
        cfg.detector.detector,
        img_rgb,
        cfg.detector.detection_size,
        cfg.detector.text_threshold,
        cfg.detector.box_threshold,
        cfg.detector.unclip_ratio,
        cfg.detector.det_invert,
        cfg.detector.det_gamma_correct,
        cfg.detector.det_rotate,
        cfg.detector.det_auto_rotate,
        device=device,
        verbose=False,
    )

    # Pick top regions by area to reduce OCR cost
    if textlines:
        textlines = sorted(textlines, key=lambda q: getattr(q, 'area', 0), reverse=True)[:8]

    probe_ocr = Ocr.ocr48px_ctc
    ocr_cfg = Config().ocr
    ocr_cfg.ocr = probe_ocr
    ocr_cfg.prob = 0.2

    ocr_out = await dispatch_ocr(probe_ocr, img_rgb, textlines, ocr_cfg, device)

    texts = []
    for r in ocr_out or []:
        t = getattr(r, 'text', '')
        if t:
            texts.append(t)
    joined = "\n".join(texts)
    detected = _detect_lang_from_text(joined)

    return {
        "detected_lang": detected,
        "probe_ocr": probe_ocr.value if hasattr(probe_ocr, 'value') else str(probe_ocr),
        "detector": detector_key,
        "sample_text": joined[:400],
    }

# --- 3. 路由定义 ---
@app.get("/results/list")
async def results_list():
    try:
        if not os.path.isdir(RESULT_ROOT):
            return {"directories": [], "items": []}

        folders = []
        for name in os.listdir(RESULT_ROOT):
            folder_path = os.path.join(RESULT_ROOT, name)
            if not os.path.isdir(folder_path):
                continue
            try:
                mtime = os.path.getmtime(folder_path)
            except Exception:
                mtime = 0.0

            cover = ""
            count = 0
            try:
                files = [
                    fn
                    for fn in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, fn))
                    and os.path.splitext(fn)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
                ]
                files.sort()
                count = len(files)
                cover = files[0] if files else ""
            except Exception:
                cover = ""
                count = 0

            title = name
            try:
                meta = _read_task_meta(folder_path)
                t = meta.get("title")
                if isinstance(t, str) and t.strip():
                    title = t.strip()
            except Exception:
                title = name

            folders.append({
                "id": name,
                "title": title,
                "updated_at": mtime,
                "cover": cover,
                "count": count,
            })

        folders.sort(key=lambda x: x.get("updated_at", 0.0), reverse=True)
        items = folders[:10]
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

    if display_title is not None:
        try:
            title = str(display_title)
            meta = _read_task_meta(folder_path)
            meta["title"] = title
            _write_task_meta(folder_path, meta)
        except Exception:
            # ignore meta write failures
            pass

    out_path = os.path.join(folder_path, safe_name)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(out_path, "wb") as f:
        f.write(data)

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

    files = [
        fn
        for fn in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, fn))
        and os.path.splitext(fn)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    ]
    files.sort()
    meta = _read_task_meta(folder_path)
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

@app.post("/probe_lang")
async def probe_lang(
    file: UploadFile = File(...),
    detector: str = Form("ctd"),
    detection_size: int = Form(1536),
):
    temp_path = None
    try:
        allowed_detectors = {"default", "ctd", "paddle"}
        if detector not in allowed_detectors:
            return {"status": "error", "message": f"Invalid detector: {detector}"}

        allowed_detection_sizes = {1024, 1536, 2048, 2560}
        if detection_size not in allowed_detection_sizes:
            return {"status": "error", "message": f"Invalid detection_size: {detection_size}"}

        temp_path = _save_upload_to_temp(file, "manga_probe")
        img_pil = Image.open(temp_path).convert('RGB')
        info = await _probe_lang_for_image(img_pil, detector_key=detector, detection_size=detection_size)
        return {"status": "success", **info}
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

@app.post("/convert_mobi_to_epub")
async def convert_mobi_to_epub(file: UploadFile = File(...)):
    temp_in = None
    temp_out = None
    try:
        temp_in = _save_upload_to_temp(file, "manga_mobi")
        temp_out = os.path.join(tempfile.gettempdir(), f"mobi_{uuid.uuid4().hex}.epub")

        import subprocess
        import shutil

        def _run_convert() -> None:
            exe = shutil.which("ebook-convert") or shutil.which("ebook-convert.exe")
            attempted = []
            if exe:
                attempted.append(exe)
            else:
                attempted_paths = [
                    r"C:\\Program Files\\Calibre2\\ebook-convert.exe",
                    r"C:\\Program Files (x86)\\Calibre2\\ebook-convert.exe",
                ]
                for p in attempted_paths:
                    attempted.append(p)
                    if os.path.exists(p):
                        exe = p
                        break

            if not exe:
                raise RuntimeError(
                    "未找到 Calibre 的 ebook-convert。\n"
                    "请先安装 Calibre，并确保 ebook-convert 在 PATH 中，或安装在默认目录。\n"
                    f"已尝试路径：{'; '.join(attempted)}"
                )

            try:
                p = subprocess.run(
                    [exe, temp_in, temp_out],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    check=False,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "ebook-convert 启动失败（系统找不到可执行文件）。\n"
                    f"exe={exe}"
                ) from e

            if p.returncode != 0 or (not os.path.exists(temp_out)):
                err = (p.stderr or p.stdout or "").strip()
                if len(err) > 2000:
                    err = err[-2000:]
                raise RuntimeError(f"MOBI 转 EPUB 失败：{err or 'unknown error'}")

        await asyncio.to_thread(_run_convert)

        with open(temp_out, "rb") as f:
            data = f.read()

        return StreamingResponse(
            BytesIO(data),
            media_type="application/epub+zip",
            headers={"Content-Disposition": "attachment; filename=converted.epub"},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        for p in (temp_in, temp_out):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/convert_rar_to_zip")
async def convert_rar_to_zip(file: UploadFile = File(...)):
    temp_in = None
    temp_out = None
    temp_dir = None
    try:
        temp_in = _save_upload_to_temp(file, "manga_rar")
        temp_dir = os.path.join(tempfile.gettempdir(), f"rar_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_out = os.path.join(tempfile.gettempdir(), f"rar_{uuid.uuid4().hex}.zip")

        import subprocess
        import shutil
        import zipfile

        def _run_convert() -> None:
            exe = shutil.which("7z") or shutil.which("7z.exe")
            attempted = []
            if exe:
                attempted.append(exe)
            else:
                attempted_paths = [
                    r"C:\\Program Files\\7-Zip\\7z.exe",
                    r"C:\\Program Files (x86)\\7-Zip\\7z.exe",
                ]
                for p in attempted_paths:
                    attempted.append(p)
                    if os.path.exists(p):
                        exe = p
                        break

            if not exe:
                raise RuntimeError(
                    "未找到 7-Zip 的 7z.exe。\n"
                    "请先安装 7-Zip，并确保 7z 在 PATH 中，或安装在默认目录。\n"
                    f"已尝试路径：{'; '.join(attempted)}"
                )

            p = subprocess.run(
                [exe, "x", "-y", f"-o{temp_dir}", temp_in],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False,
            )
            if p.returncode != 0:
                err = (p.stderr or p.stdout or "").strip()
                if len(err) > 2000:
                    err = err[-2000:]
                raise RuntimeError(f"RAR/CBR 解包失败：{err or 'unknown error'}")

            allowed_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
            extracted_files: list[str] = []
            for root, _, files in os.walk(temp_dir):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in allowed_exts:
                        extracted_files.append(os.path.join(root, fn))

            extracted_files.sort()
            if not extracted_files:
                raise RuntimeError("RAR/CBR 内未找到可翻译的图片")

            with zipfile.ZipFile(temp_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fp in extracted_files:
                    arcname = os.path.relpath(fp, temp_dir).replace(os.sep, "/")
                    zf.write(fp, arcname)

        await asyncio.to_thread(_run_convert)

        with open(temp_out, "rb") as f:
            data = f.read()

        base = os.path.splitext(os.path.basename(file.filename or "archive"))[0]
        return StreamingResponse(
            BytesIO(data),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={base}.zip"},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

    finally:
        for p in (temp_in, temp_out):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

@app.get("/mocr/status")
async def mocr_status():
    return {
        "downloaded": _mocr_is_downloaded(),
        "cache_dir": _mocr_repo_cache_dir(),
        "download_state": MOCR_DOWNLOAD.get("state", "unknown"),
        "download_error": MOCR_DOWNLOAD.get("error", ""),
        "download_endpoint": MOCR_DOWNLOAD.get("endpoint", ""),
        "download_attempts": MOCR_DOWNLOAD.get("attempts", []),
    }

@app.post("/mocr/download")
async def mocr_download():
    if _mocr_is_downloaded():
        return {
            "status": "ok",
            "downloaded": True,
            "cache_dir": _mocr_repo_cache_dir(),
        }
    if MOCR_DOWNLOAD.get("state") == "downloading":
        return {
            "status": "ok",
            "downloaded": False,
            "download_state": "downloading",
            "cache_dir": _mocr_repo_cache_dir(),
        }

    t = threading.Thread(target=_download_mocr_background, daemon=True)
    t.start()
    return {
        "status": "ok",
        "downloaded": False,
        "download_state": "downloading",
        "cache_dir": _mocr_repo_cache_dir(),
    }

@app.post("/scan")
async def scan_image(
    file: UploadFile = File(...),
    lang: str = Form("ja"),
    inpainter: str = Form("lama_mpe"),
    detector: str = Form("ctd"),
    detection_size: int = Form(1536),
    inpainting_size: int = Form(2048),
    translator: str = Form("deepseek"),
    target_lang: str = Form("CHS"),
    ocr: str = Form("auto"),
):
    global MOCR_DISABLED, MOCR_DISABLED_REASON
    temp_path = None
    try:
        allowed_inpainters = {"lama_mpe", "lama_large"}
        if inpainter not in allowed_inpainters:
            inpainter = "lama_mpe"

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

        temp_path = _save_upload_to_temp(file, "manga_scan")
        img_pil = Image.open(temp_path).convert('RGB')
        
        from manga_translator.detection import resolve_detector_key
        cfg = Config()
        cfg.detector.detector = resolve_detector_key(detector)

        cfg.detector.detection_size = detection_size
        cfg.detector.text_threshold = 0.3

        cfg.inpainter.inpainting_size = inpainting_size

        detected_lang = lang
        if lang == "auto":
            probe = await _probe_lang_for_image(img_pil, detector_key=detector, detection_size=detection_size)
            detected_lang = probe.get("detected_lang", "unknown")

        ocr_requested = ocr
        fallback_reason = ""

        # OCR 路由（源语言）
        used_ocr = Ocr.ocr48px_ctc
        if detected_lang == "ja":
            used_ocr = Ocr.mocr
            cfg.detector.det_auto_rotate = True
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

        if used_ocr == Ocr.mocr and MOCR_DISABLED:
            used_ocr = Ocr.ocr48px_ctc
            cfg.detector.det_auto_rotate = False
            fallback_reason = MOCR_DISABLED_REASON or "mocr_failed_torch_security_cached"
            print(
                f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
            )

        if used_ocr == Ocr.mocr and (not _mocr_is_downloaded()):
            used_ocr = Ocr.ocr48px_ctc
            cfg.detector.det_auto_rotate = False
            if not fallback_reason:
                fallback_reason = "mocr_not_downloaded"
            print(
                f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
            )

        if used_ocr == Ocr.mocr:
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            if os.environ.get('HF_HUB_OFFLINE') == '1':
                try:
                    del os.environ['HF_HUB_OFFLINE']
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

        print(f"正在生成翻译画面 (LAMA Inpainting)...")
        try:
            ctx = await translator_instance.translate(img_pil, cfg)
        except Exception as e:
            msg = str(e)
            if used_ocr == Ocr.mocr and (
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
                print(
                    f"OCR fallback: requested={ocr_requested} used_ocr={used_ocr.value if hasattr(used_ocr, 'value') else str(used_ocr)} reason={fallback_reason}"
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

                if clean_arr.ndim == 3 and clean_arr.shape[-1] == 3:
                    # Inpainting output is typically BGR (OpenCV); convert to RGB for correct colors.
                    clean_arr = clean_arr[..., ::-1]

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

        def _fallback_draw(img_arr: np.ndarray, bbox: tuple[int, int, int, int], text: str, fontp: str, fs: int, fg: list[int], bg: list[int]) -> np.ndarray:
            pil_img = Image.fromarray(img_arr).convert("RGB")
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(fontp, fs) if fontp else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            x0, y0, x1, y1 = bbox
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            pad = max(2, int(min(w, h) * 0.06))
            max_w = max(1, w - pad * 2)
            max_h = max(1, h - pad * 2)

            stroke_w = 0 if disable_font_border else max(1, int(fs * 0.18))

            def measure(s: str) -> int:
                try:
                    box = draw.textbbox((0, 0), s, font=font, stroke_width=stroke_w)
                    return int(box[2] - box[0])
                except Exception:
                    return int(len(s) * fs * 0.6)

            lines: list[str] = []
            for para in str(text or "").split("\n"):
                chars = list(para)
                line = ""
                for ch in chars:
                    test = line + ch
                    if line and measure(test) > max_w:
                        lines.append(line)
                        line = ch
                    else:
                        line = test
                if line:
                    lines.append(line)
                if para == "" and len(text.split("\n")) > 1:
                    lines.append("")

            line_h = int(fs * 1.15)
            total_h = len(lines) * line_h
            start_y = y0 + pad + max(0, int((max_h - total_h) / 2))
            fill = (int(fg[0]), int(fg[1]), int(fg[2]))
            stroke = (int(bg[0]), int(bg[1]), int(bg[2]))

            for i, ln in enumerate(lines):
                ln_w = measure(ln)
                xx = x0 + pad + max(0, int((max_w - ln_w) / 2))
                yy = start_y + i * line_h
                draw.text((xx, yy), ln, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)

            return np.array(pil_img)

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
                pts = []
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

            # Defaults: white text with black stroke-like bg if missing
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
                    # Empirical: font size roughly 50-65% of bubble height
                    font_size = int(max(12, min(256, h_est * 0.6)))
                except Exception:
                    font_size = 24

            direction = item.get("direction", "auto")
            alignment = item.get("alignment", "auto")

            letter_spacing = item.get("letter_spacing", 1.0)
            line_spacing_region = item.get("line_spacing", 1.0)
            try:
                letter_spacing = float(letter_spacing)
            except Exception:
                letter_spacing = 1.0
            try:
                line_spacing_region = float(line_spacing_region)
            except Exception:
                line_spacing_region = 1.0

            # TextBlock expects a list of textline polygons; for editor we treat the region polygon as a single line.
            tb = TextBlock(
                lines=[poly_i],
                texts=[str(item.get("text_original") or "")],
                font_size=font_size,
                angle=float(item.get("angle", 0) or 0),
                translation=translation,
                fg_color=tuple(fg_color[:3]),
                bg_color=tuple(bg_color[:3]),
                line_spacing=line_spacing_region,
                letter_spacing=letter_spacing,
                direction=str(direction or "auto"),
                alignment=str(alignment or "auto"),
                target_lang=str(target_lang or "CHS"),
            )

            # Render directly into the user-provided polygon (no auto-resize in this endpoint).

            xs_i = poly_i[:, 0]
            ys_i = poly_i[:, 1]
            x0 = int(max(0, min(xs_i)))
            x1 = int(min(img.shape[1], max(xs_i)))
            y0 = int(max(0, min(ys_i)))
            y1 = int(min(img.shape[0], max(ys_i)))
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                continue
            before_sum = int(img[y0:y1, x0:x1].sum())

            img = render_textblock(
                img,
                tb,
                poly_i,
                hyphenate=True,
                line_spacing=line_spacing,
                disable_font_border=bool(disable_font_border),
            )

            after_sum = int(img[y0:y1, x0:x1].sum())
            if after_sum == before_sum:
                img = _fallback_draw(img, (x0, y0, x1, y1), translation, selected_font_path, font_size, fg_color, bg_color)

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