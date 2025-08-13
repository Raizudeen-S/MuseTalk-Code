import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf  # Keep if inference_config is used, otherwise remove
from transformers import WhisperModel
import sys
import tempfile
import uuid  # To create unique temporary filenames

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

try:
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import (
        get_file_type,
        get_video_fps,
        datagen,
        load_all_model,
    )
    from musetalk.utils.preprocessing import (
        get_landmark_and_bbox,
        read_imgs,
        coord_placeholder,
    )
except ImportError:
    print("Error: Ensure the 'musetalk' library is correctly installed and accessible.")
    sys.exit(1)


FFMPEG_PATH = "/usr/bin"  # Or set via environment variable
GPU_ID = 0
VAE_TYPE = "sd-vae"
UNET_CONFIG_PATH = "./models/musetalk/config.json"
# Base model paths - these will be selected based on 'version' parameter later
UNET_MODEL_PATH_V15 = "./models/musetalkV15/unet.pth"
UNET_MODEL_PATH_V1 = "./models/musetalkV1/unet.pth"  # Assuming you have a v1 path
WHISPER_DIR = "./models/whisper"
# INFERENCE_CONFIG_PATH = "configs/inference/test_img.yaml" # Less relevant for API?
DEFAULT_BBOX_SHIFT_V1 = 0
RESULT_DIR = "./results_api"  # Directory for API outputs
DEFAULT_EXTRA_MARGIN = 10
DEFAULT_FPS = 25
DEFAULT_AUDIO_PADDING_LEFT = 2
DEFAULT_AUDIO_PADDING_RIGHT = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_PARSING_MODE = "jaw"
DEFAULT_LEFT_CHEEK_WIDTH = 90
DEFAULT_RIGHT_CHEEK_WIDTH = 90


def fast_check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            text=True,
            errors="ignore",
        )
        print("ffmpeg found in PATH.")
        return True
    except FileNotFoundError:
        print("ffmpeg not found in PATH.")
        return False
    except Exception as e:
        print(f"Error checking ffmpeg: {e}")
        return False


if not fast_check_ffmpeg():
    print(f"Attempting to add {FFMPEG_PATH} to PATH")
    path_separator = ";" if sys.platform == "win32" else ":"
    current_path = os.environ.get("PATH", "")
    if FFMPEG_PATH not in current_path:
        os.environ["PATH"] = f"{FFMPEG_PATH}{path_separator}{current_path}"
        if not fast_check_ffmpeg():
            print(
                f"Warning: Unable to find ffmpeg even after adding {FFMPEG_PATH}. Video generation may fail."
            )
            print(f"Current PATH: {os.environ['PATH']}")
        else:
            print(f"Successfully added {FFMPEG_PATH} to PATH.")
    else:
        print(f"{FFMPEG_PATH} already seems to be in PATH.")


# --- Device Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{GPU_ID}")
    print(f"Using GPU: {GPU_ID}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


MODELS_CACHE = {}  # Cache loaded models {version: {vae, unet, pe, fp}}


def get_models(version="v15", use_float16=False):
    """Loads or retrieves models from cache."""
    if version in MODELS_CACHE:
        print(f"Using cached models for version: {version}")
        return MODELS_CACHE[version]

    print(f"Loading models for version: {version}, float16: {use_float16}...")
    unet_model_path = UNET_MODEL_PATH_V15 if version == "v15" else UNET_MODEL_PATH_V1

    if not os.path.exists(unet_model_path):
        raise FileNotFoundError(
            f"UNet model path not found for version {version}: {unet_model_path}"
        )
    if not os.path.exists(UNET_CONFIG_PATH):
        raise FileNotFoundError(f"UNet config path not found: {UNET_CONFIG_PATH}")
    if not os.path.isdir(WHISPER_DIR):
        raise FileNotFoundError(f"Whisper directory not found: {WHISPER_DIR}")

    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type=VAE_TYPE,
        unet_config=UNET_CONFIG_PATH,
        device=DEVICE,
    )

    if use_float16 and DEVICE != torch.device("cpu"):
        print("Converting models to float16")
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    else:
        pe = pe.float()
        vae.vae = vae.vae.float()
        unet.model = unet.model.float()

    pe = pe.to(DEVICE)
    vae.vae = vae.vae.to(DEVICE)
    unet.model = unet.model.to(DEVICE)

    if version == "v15":
        fp = FaceParsing(
            left_cheek_width=DEFAULT_LEFT_CHEEK_WIDTH,  # Make these configurable if needed
            right_cheek_width=DEFAULT_RIGHT_CHEEK_WIDTH,
        )
    else:  # v1
        fp = FaceParsing()

    models = {
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "fp": fp,
        "timesteps": torch.tensor([0], device=DEVICE),  # Keep timesteps with models
    }
    MODELS_CACHE[version] = models
    print(f"Models loaded successfully for version: {version}")
    return models


try:
    print("Loading Whisper model...")
    _whisper_dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )  # Default to fp16 on GPU
    whisper_model = WhisperModel.from_pretrained(WHISPER_DIR)
    whisper_model = whisper_model.to(device=DEVICE, dtype=_whisper_dtype).eval()
    whisper_model.requires_grad_(False)
    print(f"Whisper model loaded to {DEVICE} with dtype {_whisper_dtype}.")

    audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    print("Audio Processor initialized.")

except Exception as e:
    print(f"Error loading Whisper model or Audio Processor: {e}")
    print("MuseTalk API cannot function without Whisper.")
    sys.exit(1)


os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()


@app.post("/generate/", response_class=FileResponse)
async def generate_video(
    video_file: UploadFile = File(
        ..., description="Source video file, image file, or a zip file of images"
    ),
    audio_file: UploadFile = File(
        ..., description="Driving audio file (e.g., wav, mp3)"
    ),
    version: str = Form("v15", enum=["v1", "v15"], description="Model version to use"),
    bbox_shift: int = Form(
        0, description="Bounding box shift value (used for v1, ignored for v15)"
    ),
    use_float16: bool = Form(
        False, description="Use float16 for faster inference (requires GPU)"
    ),
    extra_margin: int = Form(
        DEFAULT_EXTRA_MARGIN, description="Extra margin for face cropping (v15 only)"
    ),
    fps: int = Form(
        DEFAULT_FPS,
        description="Frames per second (used if input is an image or image folder)",
    ),
    batch_size: int = Form(DEFAULT_BATCH_SIZE, description="Batch size for inference"),
    output_vid_name: str = Form(
        None, description="Optional name for the output video file (without extension)"
    ),
    parsing_mode: str = Form(
        DEFAULT_PARSING_MODE, description="Face blending parsing mode (v15 only)"
    ),
    # Add cheek width params if needed for v15 customization via API
    # left_cheek_width: int = Form(DEFAULT_LEFT_CHEEK_WIDTH, description="Left cheek width (v15)"),
    # right_cheek_width: int = Form(DEFAULT_RIGHT_CHEEK_WIDTH, description="Right cheek width (v15)"),
):
    """
    Generates a talking head video based on input video/image and audio.
    """
    temp_dir = None
    input_video_path = None
    input_audio_path = None
    output_video_path = None
    extracted_images_dir = None  # For handling zip or video input

    try:
        req_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp(prefix=f"musetalk_api_{req_id}_")
        print(f"[{req_id}] Created temporary directory: {temp_dir}")

        input_video_path = os.path.join(temp_dir, f"input_{video_file.filename}")
        input_audio_path = os.path.join(temp_dir, f"input_{audio_file.filename}")

        with open(input_video_path, "wb") as f_video:
            shutil.copyfileobj(video_file.file, f_video)
        print(f"[{req_id}] Saved video file to: {input_video_path}")

        with open(input_audio_path, "wb") as f_audio:
            shutil.copyfileobj(audio_file.file, f_audio)
        print(f"[{req_id}] Saved audio file to: {input_audio_path}")

        models = get_models(version, use_float16)
        vae = models["vae"]
        unet = models["unet"]
        pe = models["pe"]
        fp = models["fp"]  # Face parser instance
        timesteps = models["timesteps"]
        unet_dtype = unet.model.dtype  # Get the actual dtype UNet is using

        current_whisper_dtype = whisper_model.dtype
        if unet_dtype != current_whisper_dtype:
            print(f"[{req_id}] Casting Whisper model to {unet_dtype} to match UNet.")
            try:
                whisper_model = whisper_model.to(dtype=unet_dtype)
            except Exception as cast_err:
                print(
                    f"[{req_id}] Warning: Could not cast Whisper model to {unet_dtype}. Dtype mismatch may cause issues. Error: {cast_err}"
                )

        file_type = get_file_type(input_video_path)
        input_img_list = []

        if file_type == "video":
            print(f"[{req_id}] Input is a video. Extracting frames...")
            extracted_images_dir = os.path.join(temp_dir, "extracted_frames")
            os.makedirs(extracted_images_dir, exist_ok=True)
            cmd = f'ffmpeg -v fatal -i "{input_video_path}" -start_number 0 "{extracted_images_dir}/%08d.png"'
            print(f"[{req_id}] Running ffmpeg command: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[{req_id}] Error extracting frames with ffmpeg: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to extract frames from video: {e}"
                )
            input_img_list = sorted(
                glob.glob(os.path.join(extracted_images_dir, "*.png"))
            )
            try:
                video_fps = get_video_fps(input_video_path)
                print(f"[{req_id}] Detected video FPS: {video_fps}")
                if video_fps and video_fps > 0:
                    fps = video_fps
                else:
                    print(
                        f"[{req_id}] Could not detect valid FPS from video, using provided/default FPS: {fps}"
                    )

            except Exception as fps_err:
                print(
                    f"[{req_id}] Warning: Could not get video FPS ({fps_err}), using provided/default FPS: {fps}"
                )

            if not input_img_list:
                raise HTTPException(
                    status_code=500, detail="ffmpeg extracted no frames from the video."
                )
            print(f"[{req_id}] Extracted {len(input_img_list)} frames.")

        elif file_type == "image":
            print(f"[{req_id}] Input is a single image.")
            input_img_list = [input_video_path]
            print(f"[{req_id}] Using specified FPS: {fps}")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input file type for video: {video_file.filename}. Must be video or image.",
            )  # Add zip later

        # --- 5. Extract Audio Features ---
        try:
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(
                input_audio_path
            )
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                DEVICE,
                unet_dtype,  # Use the UNet's dtype for consistency
                whisper_model,
                librosa_length,
                fps=fps,
                audio_padding_length_left=DEFAULT_AUDIO_PADDING_LEFT,
                audio_padding_length_right=DEFAULT_AUDIO_PADDING_RIGHT,
            )
            print(f"[{req_id}] Audio processed into {len(whisper_chunks)} chunks.")
        except Exception as e:
            print(f"[{req_id}] Error processing audio: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process audio file: {e}"
            )

        print(f"[{req_id}] Extracting landmarks and bounding boxes...")
        effective_bbox_shift = (
            0 if version == "v15" else bbox_shift
        )  # v15 ignores bbox_shift
        try:
            coord_list, frame_list = get_landmark_and_bbox(
                input_img_list, effective_bbox_shift
            )
        except Exception as e:
            print(f"[{req_id}] Error during landmark/bbox extraction: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process input images for landmarks: {e}",
            )

        if not frame_list:
            raise HTTPException(
                status_code=500, detail="Could not read or process any input frames."
            )
        print(f"[{req_id}] Processed {len(frame_list)} frames for landmarks.")

        print(f"[{req_id}] Encoding frames to latents...")
        input_latent_list = []
        valid_frame_count = 0
        for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                print(
                    f"[{req_id}] Warning: Skipping frame {i} due to missing bbox/landmark."
                )
                continue
            x1, y1, x2, y2 = bbox

            if version == "v15":
                y2_adjusted = min(y2 + extra_margin, frame.shape[0])
            else:
                y2_adjusted = y2  # No adjustment for v1

            if y1 >= y2_adjusted or x1 >= x2:
                print(
                    f"[{req_id}] Warning: Skipping frame {i} due to invalid bbox dimensions after adjustment ({x1},{y1},{x2},{y2_adjusted})."
                )
                continue

            crop_frame = frame[y1:y2_adjusted, x1:x2]

            if crop_frame.size == 0:
                print(f"[{req_id}] Warning: Skipping frame {i} due to empty crop.")
                continue

            try:
                crop_frame_resized = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                latents = vae.get_latents_for_unet(crop_frame_resized)
                input_latent_list.append(latents)
                valid_frame_count += 1
            except Exception as e:
                print(
                    f"[{req_id}] Warning: Error processing frame {i} during VAE encoding: {e}. Skipping frame."
                )

        if not input_latent_list:
            raise HTTPException(
                status_code=500,
                detail="No valid frames could be processed and encoded.",
            )
        print(f"[{req_id}] Encoded {valid_frame_count} valid frames to latents.")

        if len(input_latent_list) == 1:
            input_latent_list_cycle = [input_latent_list[0], input_latent_list[0]]
            coord_list_cycle = [coord_list[0], coord_list[0]]
            frame_list_cycle = [frame_list[0], frame_list[0]]
        else:
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            frame_list_cycle = frame_list + frame_list[::-1]

        print(f"[{req_id}] Starting batch inference...")
        video_num = len(whisper_chunks)
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,  # Keep as 0 unless specific use case
            device=DEVICE,
        )

        res_frame_list = []
        total_batches = int(np.ceil(float(video_num) / batch_size))

        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=total_batches, desc=f"[{req_id}] Inference Progress")
        ):
            whisper_batch = whisper_batch.to(DEVICE, dtype=unet_dtype)
            latent_batch = latent_batch.to(DEVICE, dtype=unet.model.dtype)

            audio_feature_batch = pe(whisper_batch)

            pred_latents = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample

            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)  # Append numpy arrays (CPU)

        if not res_frame_list:
            raise HTTPException(
                status_code=500,
                detail="Inference completed but generated no output frames.",
            )
        print(f"[{req_id}] Inference finished, generated {len(res_frame_list)} frames.")

        print(f"[{req_id}] Merging generated faces onto original frames...")
        result_img_save_path = os.path.join(temp_dir, "output_frames")
        os.makedirs(result_img_save_path, exist_ok=True)

        num_coords = len(coord_list_cycle)
        num_orig_frames = len(frame_list_cycle)

        for i, res_frame in enumerate(
            tqdm(res_frame_list, desc=f"[{req_id}] Post-processing")
        ):
            coord_idx = i % num_coords
            frame_idx = i % num_orig_frames

            bbox = coord_list_cycle[coord_idx]
            ori_frame = copy.deepcopy(frame_list_cycle[frame_idx])

            if bbox == coord_placeholder:
                print(
                    f"[{req_id}] Warning: Skipping post-processing for frame {i} due to missing bbox."
                )
                continue

            x1, y1, x2, y2 = bbox

            if version == "v15":
                y2_adjusted = min(y2 + extra_margin, ori_frame.shape[0])
            else:
                y2_adjusted = y2

            target_w = x2 - x1
            target_h = y2_adjusted - y1

            if target_w <= 0 or target_h <= 0:
                print(
                    f"[{req_id}] Warning: Skipping post-processing for frame {i} due to invalid target dimensions ({target_w}x{target_h})."
                )
                continue

            try:
                res_frame_resized = cv2.resize(
                    res_frame.astype(np.uint8), (target_w, target_h)
                )
            except Exception as e:
                print(
                    f"[{req_id}] Warning: Error resizing generated frame {i}: {e}. Skipping frame."
                )
                continue

            try:
                if version == "v15":
                    combine_frame = get_image(
                        ori_frame,
                        res_frame_resized,
                        [x1, y1, x2, y2_adjusted],
                        mode=parsing_mode,
                        fp=fp,
                    )
                else:
                    combine_frame = get_image(
                        ori_frame, res_frame_resized, [x1, y1, x2, y2], fp=fp
                    )

                cv2.imwrite(
                    os.path.join(result_img_save_path, f"{str(i).zfill(8)}.png"),
                    combine_frame,
                )
            except Exception as e:
                print(
                    f"[{req_id}] Error merging or saving frame {i}: {e}. Skipping frame."
                )

        saved_frames = glob.glob(os.path.join(result_img_save_path, "*.png"))
        if not saved_frames:
            raise HTTPException(
                status_code=500,
                detail="Post-processing finished, but no frames were successfully saved.",
            )
        print(f"[{req_id}] Saved {len(saved_frames)} merged frames.")

        input_basename = os.path.splitext(os.path.basename(input_video_path))[0]
        audio_basename = os.path.splitext(os.path.basename(input_audio_path))[0]
        if output_vid_name is None:
            final_output_basename = (
                f"{input_basename}_{audio_basename}_{version}_{req_id}.mp4"
            )
        else:
            safe_name = "".join(
                c for c in output_vid_name if c.isalnum() or c in ("_", "-")
            ).rstrip()
            final_output_basename = f"{safe_name}.mp4"

        output_video_path = os.path.join(RESULT_DIR, final_output_basename)

        cmd_img2video = f'ffmpeg -y -v warning -r {fps} -f image2 -i "{result_img_save_path}/%08d.png" -vcodec libx264 -vf "format=yuv420p" -crf 18 "{output_video_path}"'  # Output directly, overwrite if exists
        print(f"[{req_id}] Running video generation command: {cmd_img2video}")
        try:
            subprocess.run(cmd_img2video, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{req_id}] Error creating video from frames: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate video from frames: {e}"
            )

        final_video_with_audio_path = os.path.join(
            RESULT_DIR, f"final_{final_output_basename}"
        )
        cmd_combine_audio = f'ffmpeg -y -v warning -i "{output_video_path}" -i "{input_audio_path}" -c:v copy -c:a aac -shortest "{final_video_with_audio_path}"'
        print(f"[{req_id}] Running audio combination command: {cmd_combine_audio}")
        try:
            subprocess.run(cmd_combine_audio, shell=True, check=True)
            os.remove(output_video_path)  # Remove the video-only version
            output_video_path = final_video_with_audio_path  # Point to the final file
        except subprocess.CalledProcessError as e:
            print(f"[{req_id}] Error combining audio and video: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to combine audio and video: {e}"
            )

        print(f"[{req_id}] Final video saved to: {output_video_path}")

        return FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename=os.path.basename(output_video_path),
        )

    except HTTPException as e:
        raise e
    except FileNotFoundError as e:
        print(f"[{req_id}] Error: Required file or directory not found: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Server configuration error: Missing file/directory - {e}",
        )
    except Exception as e:
        print(f"[{req_id}] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()  # Log the full traceback for debugging
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )

    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[{req_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                print(
                    f"[{req_id}] Error cleaning up temporary directory {temp_dir}: {cleanup_err}"
                )


@app.route("/", methods=["GET"])
async def root():
    return {"message": "MuseTalk API is running!"}


if __name__ == "__main__":
    try:
        print("Pre-loading default models (v15)...")
        get_models(version="v15", use_float16=False)  # Load default float32 first
        if DEVICE != torch.device("cpu"):
            print("Pre-loading default models (v15 float16)...")
            get_models(
                version="v15", use_float16=True
            )  # Also cache float16 version if GPU available
    except Exception as preload_err:
        print(f"Error pre-loading models: {preload_err}")

    print("Starting FastAPI server...")
    uvicorn.run(
        "fast_api:app", host="0.0.0.0", port=8000, use_colors=True
    )  # Use the filename 'main.py'
