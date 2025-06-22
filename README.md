# Comic Strip Generator

A tool for automatically generating comic-style frames from videos using AI-powered keyframe extraction and cartoon-style effects.

## Features

- Video keyframe extraction using deep learning
- Cartoon-style effect application
- Web interface for easy use
- Support for multiple video formats
- Automatic cleanup of old files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automated_comic_strip_generator.git
cd automated_comic_strip_generator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Start the web application:
```bash
python src/app.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

3. Upload a video file (supported formats: MP4, AVI, MOV, MKV)
4. Wait for the processing to complete
5. View the generated comic frames

## Project Structure

```
automated_comic_strip_generator/
├── data/                      # Data storage
│   ├── uploaded_videos/       # Temporary storage for uploaded videos
│   ├── extracted_frames/      # Extracted keyframes
│   └── cartoon_frames/        # Cartoonified frames
├── src/
│   ├── comic_generator/       # Main package
│   │   ├── __init__.py
│   │   ├── keyframe.py       # Keyframe extraction
│   │   ├── model.py          # Neural network models
│   │   └── style.py          # Cartoon style effects
│   └── app.py                # Flask web application
├── models/                    # Pre-trained models
├── setup.py                  # Package setup
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch
- Flask
- Other dependencies listed in requirements.txt

## Development

To contribute to the project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the pre-trained models
- OpenCV community for image processing tools 