import csv
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image as PILImage


DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
USERS_CSV = "users.csv"
ATTENDANCE_CSV = "attendance.csv"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")


def ensure_setup():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TRAINER_DIR, exist_ok=True)
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Name"])


def has_face_module():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def create_recognizer():
    if not has_face_module():
        raise RuntimeError(
            "OpenCV face module is not available. Install opencv-contrib-python."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def create_detector():
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Failed to load the Haar cascade for face detection.")
    return detector


def open_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check that it is connected and free."
        )
    return cam


def apply_nms(boxes, scores, threshold=0.3):
    if not boxes:
        return []

    converted_boxes = []
    for x1, y1, x2, y2 in boxes:
        converted_boxes.append(
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        )

    normalized_scores = [float(score) for score in scores]

    indices = cv2.dnn.NMSBoxes(
        converted_boxes, normalized_scores, threshold, threshold
    )
    if indices is None or len(indices) == 0:
        return []

    filtered_boxes = []
    for index in np.array(indices).flatten():
        filtered_boxes.append(boxes[int(index)])
    return filtered_boxes


def load_user_names():
    ensure_setup()
    user_dict = {}
    with open(USERS_CSV, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                user_dict[int(row["ID"])] = row["Name"]
            except (TypeError, ValueError, KeyError):
                continue
    return user_dict


def mark_attendance(user_id, name):
    with open(ATTENDANCE_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def collect_data():
    ensure_setup()

    user_id = simpledialog.askinteger("Input", "Enter User ID:")
    user_name = simpledialog.askstring("Input", "Enter Name:")
    if user_id is None or not user_name:
        messagebox.showwarning("Warning", "User ID and Name are required.")
        return

    user_dict = load_user_names()
    if user_id in user_dict:
        messagebox.showerror("Error", f"User ID {user_id} is already registered.")
        return

    recognizer = None
    if os.path.exists(TRAINER_FILE):
        try:
            recognizer = create_recognizer()
            recognizer.read(TRAINER_FILE)
        except RuntimeError as error:
            messagebox.showerror("Error", str(error))
            return

    try:
        cam = open_camera()
        detector = create_detector()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return

    with open(USERS_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name])

    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        face_boxes = []
        scores = []

        for (x, y, w, h) in faces:
            face_boxes.append([x, y, x + w, y + h])
            scores.append(w * h)

        filtered_faces = apply_nms(face_boxes, scores)

        for (x, y, x2, y2) in filtered_faces:
            face_img = gray[y:y2, x:x2]

            if recognizer is not None:
                try:
                    predicted_id, confidence = recognizer.predict(face_img)
                    if confidence < 50:
                        messagebox.showerror(
                            "Error",
                            f"This face is already registered as User {predicted_id}.",
                        )
                        cam.release()
                        cv2.destroyAllWindows()
                        return
                except cv2.error:
                    pass

            count += 1
            cv2.imwrite(
                os.path.join(DATASET_DIR, f"User.{user_id}.{count}.jpg"),
                face_img,
            )
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Collecting Faces - Press Q to Stop", img)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 5:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Collected {count} face samples for {user_name}")


def train_model():
    ensure_setup()

    try:
        recognizer = create_recognizer()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return

    try:
        detector = create_detector()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return
    face_samples = []
    ids = []

    for filename in os.listdir(DATASET_DIR):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(DATASET_DIR, filename)
        img = PILImage.open(img_path).convert("L")
        img_np = np.array(img, "uint8")
        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y : y + h, x : x + w])
            try:
                ids.append(int(filename.split(".")[1]))
            except (IndexError, ValueError):
                face_samples.pop()

    if not face_samples:
        messagebox.showerror("Error", "No faces found. Please register first.")
        return

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(TRAINER_FILE)
    messagebox.showinfo("Info", f"Model trained with {len(set(ids))} users.")


def recognize_faces():
    ensure_setup()

    if not os.path.exists(TRAINER_FILE):
        messagebox.showerror(
            "Error", "Trainer file not found. Please train the model first."
        )
        return

    try:
        recognizer = create_recognizer()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return

    recognizer.read(TRAINER_FILE)
    user_dict = load_user_names()
    try:
        face_cascade = create_detector()
        cam = open_camera()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return

    recognized_ids = set()

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            predicted_id, confidence = recognizer.predict(roi)

            if confidence < 65 and predicted_id in user_dict:
                name = user_dict[predicted_id]
                label = f"{name} ({predicted_id})"
                if predicted_id not in recognized_ids:
                    mark_attendance(predicted_id, name)
                    recognized_ids.add(predicted_id)
            else:
                label = "Unknown"

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Recognizing Faces - Press Q to Quit", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Recognition complete!")


def main_menu():
    root = Tk()
    root.title("Face Recognition System with Attendance")
    root.geometry("400x400")
    root.configure(bg="#e6f2ff")

    Label(
        root,
        text="Face Recognition System",
        font=("Arial", 18, "bold"),
        bg="#e6f2ff",
    ).pack(pady=20)
    Button(
        root,
        text="Register Face",
        command=collect_data,
        width=30,
        height=2,
        bg="#007acc",
        fg="white",
    ).pack(pady=10)
    Button(
        root,
        text="Train Model",
        command=train_model,
        width=30,
        height=2,
        bg="#007acc",
        fg="white",
    ).pack(pady=10)
    Button(
        root,
        text="Recognize Face",
        command=recognize_faces,
        width=30,
        height=2,
        bg="#007acc",
        fg="white",
    ).pack(pady=10)
    Button(
        root,
        text="Exit",
        command=root.quit,
        width=30,
        height=2,
        bg="#ff4d4d",
        fg="white",
    ).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    ensure_setup()
    main_menu()
