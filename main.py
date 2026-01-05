import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from typing import List, Dict, Any

from manga_translator.detection import dispatch as dispatch_detection
from manga_translator.ocr import dispatch as dispatch_ocr
from manga_translator.translators import (
    dispatch as dispatch_translation,
    TranslatorChain,
    LanguageUnsupportedException
)
from manga_translator.config import (
    TranslatorConfig,
    DetectorConfig,
    OcrConfig,
    Translator
)
from manga_translator.utils import Context

app = FastAPI()

# Initialize configs with default values
translator_config = TranslatorConfig()
detector_config = DetectorConfig()
ocr_config = OcrConfig()

# Set target language to Chinese Simplified
translator_config.target_lang = 'CHS'

# Initialize device (use GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

async def process_image(image_data: bytes) -> Dict[str, Any]:
    """Process image through detection, OCR, and translation pipeline."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        
        # 1. Run detection to find text regions
        text_regions, mask, lines_im = await dispatch_detection(
            detector_config.detector,
            image_np,
            detector_config.detection_size,
            detector_config.text_threshold,
            detector_config.box_threshold,
            detector_config.unclip_ratio,
            detector_config.det_invert,
            detector_config.det_gamma_correct,
            detector_config.det_rotate,
            detector_config.det_auto_rotate,
            device=device,
            verbose=False,
        )
        
        if not text_regions:
            return {"error": "No text regions detected"}
        
        # 2. Run OCR on detected regions
        ocr_results = await dispatch_ocr(
            'manga_ocr',  # or your preferred OCR
            image_np,
            text_regions,
            ocr_config,
            device=device
        )
        
        # 3. Extract text and bounding boxes
        regions = []
        texts_to_translate = []
        
        for i, result in enumerate(ocr_results):
            if hasattr(result, 'get_text'):  # TextBlock object
                text = result.get_text()
                box = result.xyxy  # [x1, y1, x2, y2]
                fg_color = result.fg_rgb if hasattr(result, 'fg_rgb') else [0, 0, 0]
                bg_color = result.bg_rgb if hasattr(result, 'bg_rgb') else [255, 255, 255]
            else:  # Quadrilateral or other format
                text = str(result)
                box = [0, 0, image.width, image.height]  # Fallback to full image
                fg_color = [0, 0, 0]
                bg_color = [255, 255, 255]
            
            if text.strip():
                regions.append({
                    "box": box,
                    "text_original": text,
                    "fg_color": fg_color,
                    "bg_color": bg_color
                })
                texts_to_translate.append(text)
        
        # 4. Translate the extracted text
        if texts_to_translate:
            try:
                translations = await dispatch_translation(
                    translator_config.translator_chain or 
                    TranslatorChain(f'{translator_config.translator}:{translator_config.target_lang}'),
                    texts_to_translate,
                    translator_config=translator_config,
                    device=device
                )
                
                # Update regions with translations
                for i, translation in enumerate(translations):
                    if i < len(regions):
                        regions[i]["text_translated"] = translation
            
            except LanguageUnsupportedException as e:
                return {"error": f"Translation error: {str(e)}"}
            except Exception as e:
                return {"error": f"Translation failed: {str(e)}"}
        
        return {
            "image_size": [image.width, image.height],
            "regions": regions
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.post("/scan")
async def scan_image(file: UploadFile):
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    try:
        image_data = await file.read()
        result = await process_image(image_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)