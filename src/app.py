from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from comic_generator import generate_keyframes, cartoonify, read_image
import cv2
import numpy as np
import logging
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial
import threading
from queue import Queue
import time
from threading import Thread

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates',
           static_url_path='/static')

# Configure application settings
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploaded_videos')
    EXTRACTED_FRAMES_FOLDER = os.path.join(BASE_DIR, 'data', 'extracted_frames')
    CARTOON_FRAMES_FOLDER = os.path.join(BASE_DIR, 'data', 'cartoon_frames')
    LOGS_FOLDER = os.path.join(BASE_DIR, 'logs')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # Max 500MB upload size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config.from_object(Config)

# Create necessary directories
for folder in [Config.UPLOAD_FOLDER, Config.EXTRACTED_FRAMES_FOLDER, 
               Config.CARTOON_FRAMES_FOLDER, Config.LOGS_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Ensured directory exists: {folder}")
    except Exception as e:
        logger.error(f"Failed to create directory {folder}: {str(e)}")
        raise

# Get the number of CPU cores for optimal parallel processing
CPU_CORES = multiprocessing.cpu_count()
# Use 75% of available cores to avoid system overload
WORKER_CORES = max(1, int(CPU_CORES * 0.75))

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def cleanup_previous_files():
    """Clean up files from previous processing"""
    try:
        for directory in [Config.EXTRACTED_FRAMES_FOLDER, Config.CARTOON_FRAMES_FOLDER]:
            clean_directory(directory)
        logger.info("Cleaned up previous processing files")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

def process_frame_chunk(frame_paths, output_folder):
    """Process a chunk of frames in parallel"""
    processed = []
    for frame_path in frame_paths:
        try:
            frame_file = os.path.basename(frame_path)
            with open(frame_path, 'rb') as f:
                image_data = f.read()
            img = read_image(image_data)
            if img is None:
                logger.error(f"Failed to read image: {frame_path}")
                continue
            cartoon_img = cartoonify(img)
            if cartoon_img is None:
                logger.error(f"Failed to cartoonify image: {frame_path}")
                continue
            output_path = os.path.join(output_folder, f'cartoon_{frame_file}')
            success = cv2.imwrite(output_path, cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2BGR))
            if not success:
                logger.error(f"Failed to save cartoonified image: {output_path}")
                continue
            processed.append(frame_file)
            logger.debug(f"Successfully processed frame: {frame_file}")
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {str(e)}")
    return processed

def process_frames_to_cartoon(frames_folder, output_folder):
    """Convert extracted frames to cartoon style using parallel processing"""
    try:
        frame_files = [f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))]
        if not frame_files:
            logger.warning(f"No frames found in {frames_folder}")
            return []

        # Get full paths for all frames
        frame_paths = [os.path.join(frames_folder, f) for f in frame_files]
        
        # Calculate chunk size based on number of frames and available cores
        chunk_size = max(1, len(frame_paths) // WORKER_CORES)
        frame_chunks = [frame_paths[i:i + chunk_size] for i in range(0, len(frame_paths), chunk_size)]

        processed_frames = []
        logger.info(f"Processing {len(frame_files)} frames using {WORKER_CORES} cores")
        
        # Process chunks in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=WORKER_CORES) as executor:
            chunk_results = list(executor.map(
                partial(process_frame_chunk, output_folder=output_folder),
                frame_chunks
            ))
            
            # Flatten results
            for chunk_result in chunk_results:
                if chunk_result:  # Only extend if we have results
                    processed_frames.extend(chunk_result)

        if not processed_frames:
            logger.warning("No frames were successfully processed")
        else:
            logger.info(f"Successfully processed {len(processed_frames)} frames")

        return processed_frames

    except Exception as e:
        logger.error(f"Error in parallel frame processing: {str(e)}")
        raise

def clean_directory(directory):
    """Remove all files from the specified directory"""
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return
            
        files_removed = 0
        for filename in os.listdir(directory):
            try:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_removed += 1
                    logger.info(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {filename}: {str(e)}")
                
        logger.info(f"Cleaned directory {directory}: removed {files_removed} files")
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {str(e)}")
        raise

class AsyncFrameProcessor:
    """Handles asynchronous frame processing and progress tracking"""
    def __init__(self):
        self.processed_frames = []
        self.total_frames = 0
        self.current_progress = 0
        self.processing_complete = False
        self.error_message = None
        self.lock = threading.Lock()
        self._start_time = None

    def process_video_async(self, video_path, use_gpu):
        """Start asynchronous video processing"""
        self.processing_complete = False
        self.processed_frames = []
        self.current_progress = 0
        self.error_message = None
        self._start_time = time.time()
        
        thread = Thread(target=self._process_video, args=(video_path, use_gpu))
        thread.daemon = True
        thread.start()
        return thread

    def _process_video(self, video_path, use_gpu):
        """Internal method to handle video processing"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Generate keyframes
            logger.info("Starting keyframe generation...")
            try:
                self.processed_frames = generate_keyframes(video_path, gpu=use_gpu)
                if not self.processed_frames:
                    raise Exception("No frames were generated")
                self.processing_complete = True
            except Exception as e:
                raise Exception(f"Error generating keyframes: {str(e)}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in async processing: {error_msg}")
            with self.lock:
                self.error_message = error_msg
                self.processing_complete = True

    def get_progress(self):
        """Get current processing progress"""
        with self.lock:
            return {
                'complete': self.processing_complete,
                'processed_frames': len(self.processed_frames),
                'frames': self.processed_frames if self.processing_complete else [],
                'processing_time': time.time() - self._start_time if self._start_time else 0,
                'error': self.error_message
            }

# Create global frame processor instance
frame_processor = AsyncFrameProcessor()

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        cleanup_previous_files()
        
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            logger.info(f"New video saved: {filename}")
            
            logger.info("Starting async video processing...")
            frame_processor.process_video_async(video_path, use_gpu=False)
            
            return jsonify({'message': 'Video uploaded successfully. Processing started.'}), 202
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_processing_progress():
    """Get current processing progress"""
    try:
        progress = frame_processor.get_progress()
        return jsonify(progress)
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/frames/list')
def list_frames():
    """Return a list of available frames"""
    try:
        progress = frame_processor.get_progress()
        if progress['complete'] and not progress['error']:
            return jsonify(progress['frames'])
        return jsonify([])
    except Exception as e:
        logger.error(f"Error listing frames: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/frames/<frame_type>/<frame_name>')
def get_frame(frame_type, frame_name):
    """Serve frame images"""
    try:
        if frame_type == 'original':
            return send_from_directory(Config.EXTRACTED_FRAMES_FOLDER, frame_name)
        elif frame_type == 'cartoon':
            return send_from_directory(Config.CARTOON_FRAMES_FOLDER, frame_name)
        else:
            return jsonify({'error': 'Invalid frame type'}), 400
    except Exception as e:
        logger.error(f"Error serving frame: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
