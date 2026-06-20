# Face Recognition Attendance System

A desktop attendance application built with Python, OpenCV, and Tkinter. The app captures face samples, trains an LBPH face recognition model, recognizes users through a webcam, and stores attendance records in a CSV file.

## Features

- Register users with a unique ID and name
- Capture face samples from a webcam
- Train a local face recognition model
- Recognize registered users in real time
- Save attendance with date and time in `attendance.csv`
- Simple Tkinter GUI for easy usage
- Basic runtime validation for camera access, trainer file availability, and OpenCV face-module support

## Tech Stack

- Python
- OpenCV (`opencv-contrib-python`)
- NumPy
- Pillow
- Tkinter

## Requirements

- Python 3
- Webcam
- Windows, Linux, or macOS with camera access

## Installation

```powershell
pip install -r requirements.txt
```

## Run The Project

```powershell
python python.py
```

## How It Works

1. Click `Register Face`
2. Enter a new user ID and user name
3. Allow the app to capture face samples
4. Click `Train Model` to build the recognizer
5. Click `Recognize Face` to start live recognition
6. Attendance entries are saved automatically to `attendance.csv`

## Output Files

- `python.py` - main application
- `dataset/` - stored face sample images
- `trainer/` - trained recognizer model
- `users.csv` - registered user IDs and names
- `attendance.csv` - attendance records with timestamps

## Sample Workflow

1. Register a user
2. Train the model
3. Start recognition
4. Review the saved attendance log

## Notes

- Generated files and personal data are excluded by `.gitignore`
- If `cv2.face` is missing, install `opencv-contrib-python` instead of `opencv-python`
- Make sure no other application is using the webcam while the app is running

## Future Improvements

- Export attendance to Excel
- Add admin login or password protection
- Improve duplicate-user detection
- Store data in a database instead of CSV
- Add a better user interface
