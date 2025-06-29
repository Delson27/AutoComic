import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import srt
import logging
from .model import DSN
from ..styling.effects import read_image, cartoonify
import torch.nn as nn
import torch.nn.functional as F
import shutil
from .extract_frames import extract_frames

logger = logging.getLogger(__name__)

def get_device(gpu=True):
    """Determine the appropriate device based on availability and user preference"""
    if gpu and torch.cuda.is_available():
        try:
            # Test GPU memory allocation
            torch.cuda.empty_cache()
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return torch.device('cuda')
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
    return torch.device('cpu')

def _get_features(frames, gpu=True, batch_size=32):
    """Extract features from frames using GoogLeNet with batched processing"""
    try:
        device = get_device(gpu)
        logger.info(f"Using device: {device}")

        # Load pre-trained GoogLeNet model
        googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT')
        googlenet = torch.nn.Sequential(*(list(googlenet.children())[:-1]))
        googlenet.eval()
        googlenet.to(device)

        features = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Process frames in batches
        batch_frames = []
        for frame_path in frames:
            input_image = Image.open(frame_path).convert('RGB')
            input_tensor = preprocess(input_image)
            batch_frames.append(input_tensor)
            
            if len(batch_frames) == batch_size:
                batch_tensor = torch.stack(batch_frames).to(device)
                with torch.no_grad():
                    batch_output = googlenet(batch_tensor)
                    # Reshape output to 2D: (batch_size, features)
                    batch_output = batch_output.view(batch_output.size(0), -1)
                features.extend(batch_output.cpu().numpy())
                batch_frames = []
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Process remaining frames
        if batch_frames:
            batch_tensor = torch.stack(batch_frames).to(device)
            with torch.no_grad():
                batch_output = googlenet(batch_tensor)
                # Reshape output to 2D: (batch_size, features)
                batch_output = batch_output.view(batch_output.size(0), -1)
            features.extend(batch_output.cpu().numpy())

        return np.array(features, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise

def _get_probs(features, gpu=True, mode=0):
    """Get probability scores for features with improved device handling"""
    try:
        device = get_device(gpu)
        logger.info(f"Using device: {device} for probability calculation")

        model_path = os.path.join("src", "comic_generator", "keyframes", "models", f"model_{mode}.pth.tar")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = DSN(in_dim=1024, hid_dim=256, num_layers=1, cell="lstm")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        seq = torch.from_numpy(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = model(seq)
            probs = probs.cpu().squeeze().numpy()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return probs

    except Exception as e:
        logger.error(f"Error in probability calculation: {str(e)}")
        raise

def copy_and_rename_file(src_path, dest_folder, new_name):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, new_name)
    shutil.copy(src_path, dest_path)

def generate_keyframes(video, srt_path=None, gpu=False):
    """Generate keyframes from video with improved error handling and memory management"""
    try:
        device = get_device(gpu)
        logger.info(f"Initializing keyframe generation using device: {device}")

        extracted_frames_folder = os.path.join("data", "extracted_frames")
        cartoon_frames_folder = os.path.join("data", "cartoon_frames")
        os.makedirs(cartoon_frames_folder, exist_ok=True)
        os.makedirs(extracted_frames_folder, exist_ok=True)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Process segments
        if srt_path is not None and os.path.exists(srt_path):
            with open(srt_path) as f:
                data = f.read()
            subs = srt.parse(data)
            segments = [(sub.start.total_seconds(), sub.end.total_seconds(), sub.index) for sub in subs]
        else:
            segment_length = 5
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            segments = [(start, min(start + segment_length, duration), idx) 
                       for idx, start in enumerate(range(0, int(duration), segment_length))]

        processed_frames = []
        for start_time, end_time, idx in segments:
            try:
                frames = extract_frames(video, extracted_frames_folder, start_time, end_time, 3)
                features = _get_features(frames, gpu=gpu)
                highlight_scores = _get_probs(features, gpu=gpu)
                
                sorted_indices = [i[0] for i in sorted(enumerate(highlight_scores), key=lambda x: x[1])]
                selected_keyframe = sorted_indices[-1]
                selected_frame_path = frames[selected_keyframe]

                # Save the selected keyframe
                frame_name = f"frame{idx:03d}.png"
                final_frame_path = os.path.join(extracted_frames_folder, frame_name)
                copy_and_rename_file(selected_frame_path, extracted_frames_folder, frame_name)
                
                # Apply cartoon effect
                img = read_image(final_frame_path)
                cartoon_img = cartoonify(img)
                cartoon_frame_path = os.path.join(cartoon_frames_folder, f"cartoon_{frame_name}")
                cv2.imwrite(cartoon_frame_path, cartoon_img)
                
                processed_frames.append(frame_name)

                # Clean up temporary frames
                for frame in frames:
                    try:
                        if os.path.exists(frame):
                            os.remove(frame)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary frame {frame}: {e}")

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing segment {idx}: {str(e)}")
                continue

        return processed_frames

    except Exception as e:
        logger.error(f"Error in keyframe generation: {str(e)}")
        raise
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def get_black_bar_coordinates(image_path):
    """
    Detects black bars in the image and returns the coordinates for cropping.
    Returns (x, y, w, h) for cropping.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def crop_image(image_path, x, y, w, h):
    image = cv2.imread(image_path)
    crop = image[y:y+h, x:x+w]
    cv2.imwrite(image_path, crop)

def black_bar_crop():
    ref_img_path = os.path.join("data", "extracted_frames", "frame001.png")
    x, y, w, h = get_black_bar_coordinates(ref_img_path)
    folder_dir = os.path.join("data", "extracted_frames")
    for image_name in os.listdir(folder_dir):
        img_path = os.path.join(folder_dir, image_name)
        crop_image(img_path, x, y, w, h)
    return x, y, w, h
