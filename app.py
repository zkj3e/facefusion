# app.py
import os
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ---- 引入 facefusion ----
from facefusion import state_manager, logger
from facefusion.args import apply_args
from facefusion.jobs import job_manager, job_store
from facefusion.setup_args import setup_args, gen_faceswapper_model_args
from facefusion.core import process_headless
from enum import Enum
from fastapi import Query

class FaceSwapperModel(str, Enum):
    """
    人脸替换模型（Face Swapper Model）
    - inswapper_128：小尺寸模型，速度快
    - inswapper_128_fp16：FP16 版本，显存占用低
    - hyperswap_1a_256：HyperSwap 快速模型（默认）
    - hyperswap_1b_256：改进肤色与光照
    - hyperswap_1c_256：HyperSwap 稳定高质量版本
    """
    inswapper_128 = "inswapper_128"
    inswapper_128_fp16 = "inswapper_128_fp16"
    hyperswap_1a_256 = "hyperswap_1a_256"
    hyperswap_1b_256 = "hyperswap_1b_256"
    hyperswap_1c_256 = "hyperswap_1c_256"




# =============== FastAPI ===============
app = FastAPI(title="Smart Image AI API")

# 创建临时目录
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)


# -------- 启动初始化 --------
@app.on_event("startup")
def startup():
    print("🔥 Initializing Smart Image ...")

    defaults = setup_args()
    job_keys = list(defaults.keys())
    step_keys = list(defaults.keys())
    try:
        job_store.register_job_keys(job_keys)
        job_store.register_step_keys(step_keys)
        apply_args(defaults, state_manager.init_item)
    except:
        pass

    try:
        logger.init(state_manager.get_item("log_level"))
    except:
        logger.init("info")

    try:
        job_manager.init_jobs(state_manager.get_item("jobs_path"))
    except:
        default_path = os.path.abspath("./jobs")
        os.makedirs(default_path, exist_ok=True)
        state_manager.init_item("jobs_path", default_path)
        job_manager.init_jobs(default_path)

    print("✅ Smart Image Initialized!")


# -------- 执行接口（文件上传版） --------
@app.post("/faceswap", summary="人脸替换接口", description="上传源图片和目标图片，返回换脸后的图片")
async def run_faceswap(
    source_file: UploadFile = File(..., description="源图片（人脸）"),
    target_file: UploadFile = File(..., description="目标图片（要换脸的对象）"),
    face_swapper_model: FaceSwapperModel = Query(
        FaceSwapperModel.hyperswap_1a_256,
        title="人脸替换模型",
        description=(
            "选择人脸替换使用的模型。\n\n"
            "推荐：\n"
            "- 快速：inswapper_128 / inswapper_128_fp16\n"
            "- 常规：hyperswap_1a_256 \n"
            "- 高质量： hyperswap_1c_256\n"
        )
    )
):
    """
    上传源图片和目标图片，返回换脸后的图片
    """
    # 验证文件类型
    allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    source_ext = os.path.splitext(source_file.filename)[1].lower()
    target_ext = os.path.splitext(target_file.filename)[1].lower()
    
    if source_ext not in allowed_image_extensions:
        raise HTTPException(400, f"源图片格式不支持，请使用: {', '.join(allowed_image_extensions)}")
    
    if target_ext not in allowed_image_extensions:
        raise HTTPException(400, f"目标图片格式不支持，请使用: {', '.join(allowed_image_extensions)}")
    
    # 生成唯一文件名
    source_filename = f"source_{uuid.uuid4().hex}{source_ext}"
    target_filename = f"target_{uuid.uuid4().hex}{target_ext}"
    output_filename = f"output_{uuid.uuid4().hex}{target_ext}"
    
    source_path = TEMP_DIR / source_filename
    target_path = TEMP_DIR / target_filename
    output_path = TEMP_DIR / output_filename
    
    # 保存上传的文件
    try:
        source_content = await source_file.read()
        target_content = await target_file.read()
        
        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(target_path, "wb") as f:
            f.write(target_content)
    except Exception as e:
        raise HTTPException(500, f"保存文件失败: {str(e)}")
    
    # 解析选项
    # option_dict = {}
    # if options:
    #     try:
    #         import json
    #         option_dict = json.loads(options)
    #     except:
    #         pass
    
    # 准备参数
    args = setup_args()
    args["command"] = "headless-run"
    args["source_paths"] = [str(source_path)]
    args["target_path"] = str(target_path)
    args["output_path"] = str(output_path)

    face_swapper_args = gen_faceswapper_model_args(face_swapper_model.value)
    args.update(face_swapper_args)
    
    # # 应用用户选项
    # for k, v in option_dict.items():
    #     if k in args:  # 只覆盖存在的参数
    #         args[k] = v
    
    try:
        # 初始化并执行
        apply_args(args, state_manager.init_item)
        result = process_headless(args)
        
        # 检查输出文件是否存在
        if not os.path.exists(output_path):
            # 清理临时文件
            try:
                source_path.unlink(missing_ok=True)
                target_path.unlink(missing_ok=True)
            except:
                pass
            
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "换脸失败，输出文件未生成",
                    "detail": str(result)
                }
            )
        
        # 读取输出文件
        with open(output_path, "rb") as f:
            output_content = f.read()
        
        # 清理临时文件
        try:
            source_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except:
            pass
        
        # 返回图片
        return StreamingResponse(
            BytesIO(output_content),
            media_type=f"image/{target_ext[1:].lower() if target_ext[1:] != 'jpg' else 'jpeg'}",
            headers={
                "Content-Disposition": f"attachment; filename=facefusion_result{target_ext}"
            }
        )
        
    except Exception as e:
        # 清理临时文件
        try:
            source_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except:
            pass
        
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": f"换脸处理失败: {str(e)}",
                "traceback": tb
            }
        )


# -------- 健康检查 --------
@app.get("/health")
def health():
    return {"status": "ok", "service": "faceswap-api"}


# -------- 主入口 --------
if __name__ == "__main__":
    print("🚀 Starting faceswap API on 0.0.0.0:6006 ...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=6006,
        reload=False
    )