import csv
import os
import sys
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
FACE_SAMPLE_TARGET = 5


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
        converted_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

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


def count_dataset_images():
    if not os.path.exists(DATASET_DIR):
        return 0
    return len([name for name in os.listdir(DATASET_DIR) if name.endswith(".jpg")])


def count_registered_users():
    return len(load_user_names())


def count_attendance_entries():
    if not os.path.exists(ATTENDANCE_CSV):
        return 0

    with open(ATTENDANCE_CSV, "r", newline="") as file:
        rows = [row for row in csv.reader(file) if row]
    return len(rows)


def get_last_attendance_entry():
    if not os.path.exists(ATTENDANCE_CSV):
        return "No attendance yet"

    with open(ATTENDANCE_CSV, "r", newline="") as file:
        rows = [row for row in csv.reader(file) if row]

    if not rows:
        return "No attendance yet"

    last_row = rows[-1]
    if len(last_row) < 3:
        return "Latest attendance recorded"
    return f"{last_row[1]} at {last_row[2]}"


def collect_data():
    ensure_setup()

    user_id = simpledialog.askinteger("Input", "Enter User ID:")
    user_name = simpledialog.askstring("Input", "Enter Name:")
    if user_id is None or not user_name:
        messagebox.showwarning("Warning", "User ID and Name are required.")
        return False

    user_dict = load_user_names()
    if user_id in user_dict:
        messagebox.showerror("Error", f"User ID {user_id} is already registered.")
        return False

    recognizer = None
    if os.path.exists(TRAINER_FILE):
        try:
            recognizer = create_recognizer()
            recognizer.read(TRAINER_FILE)
        except RuntimeError as error:
            messagebox.showerror("Error", str(error))
            return False

    try:
        cam = open_camera()
        detector = create_detector()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return False

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
                        return False
                except cv2.error:
                    pass

            count += 1
            cv2.imwrite(
                os.path.join(DATASET_DIR, f"User.{user_id}.{count}.jpg"),
                face_img,
            )
            cv2.rectangle(img, (x, y), (x2, y2), (20, 184, 166), 2)
            cv2.putText(
                img,
                f"Samples: {count}/{FACE_SAMPLE_TARGET}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Collecting Faces - Press Q to Stop", img)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= FACE_SAMPLE_TARGET:
            break

    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        messagebox.showwarning("Warning", "No face samples were captured.")
        return False

    with open(USERS_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name])

    messagebox.showinfo("Info", f"Collected {count} face samples for {user_name}")
    return True


def train_model():
    ensure_setup()

    try:
        recognizer = create_recognizer()
        detector = create_detector()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return False

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
            try:
                user_id = int(filename.split(".")[1])
            except (IndexError, ValueError):
                continue
            face_samples.append(img_np[y : y + h, x : x + w])
            ids.append(user_id)

    if not face_samples:
        messagebox.showerror("Error", "No faces found. Please register first.")
        return False

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(TRAINER_FILE)
    messagebox.showinfo("Info", f"Model trained with {len(set(ids))} users.")
    return True


def recognize_faces():
    ensure_setup()

    if not os.path.exists(TRAINER_FILE):
        messagebox.showerror(
            "Error", "Trainer file not found. Please train the model first."
        )
        return False

    try:
        recognizer = create_recognizer()
        face_cascade = create_detector()
        cam = open_camera()
    except RuntimeError as error:
        messagebox.showerror("Error", str(error))
        return False

    recognizer.read(TRAINER_FILE)
    user_dict = load_user_names()
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
                box_color = (46, 204, 113)
                if predicted_id not in recognized_ids:
                    mark_attendance(predicted_id, name)
                    recognized_ids.add(predicted_id)
            else:
                label = "Unknown"
                box_color = (52, 73, 94)

            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
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
    return True


def show_attendance_preview():
    preview = Toplevel()
    preview.title("Attendance Preview")
    preview.geometry("640x360")
    preview.configure(bg="#f7f1e3")

    header = Label(
        preview,
        text="Recent Attendance",
        font=("Segoe UI", 18, "bold"),
        bg="#f7f1e3",
        fg="#1f2d3d",
    )
    header.pack(pady=(18, 8))

    text_box = Text(
        preview,
        width=72,
        height=14,
        bg="#fffdf7",
        fg="#22313f",
        font=("Consolas", 11),
        relief=FLAT,
        padx=14,
        pady=14,
    )
    text_box.pack(padx=20, pady=(0, 18), fill=BOTH, expand=True)

    if not os.path.exists(ATTENDANCE_CSV):
        text_box.insert(END, "No attendance file found yet.")
    else:
        with open(ATTENDANCE_CSV, "r", newline="") as file:
            rows = [row for row in csv.reader(file) if row]

        if not rows:
            text_box.insert(END, "No attendance records found.")
        else:
            for row in rows[-10:]:
                text_box.insert(END, " | ".join(row) + "\n")

    text_box.configure(state=DISABLED)


def open_project_folder():
    folder_path = os.path.abspath(".")
    if sys.platform.startswith("win"):
        os.startfile(folder_path)
    else:
        messagebox.showinfo("Info", f"Project folder: {folder_path}")


def build_stat_card(parent, title, value_var, accent):
    card = Frame(parent, bg="#ffffff", bd=0, highlightthickness=0)
    card.pack(side=LEFT, expand=True, fill=BOTH, padx=8)

    accent_bar = Frame(card, bg=accent, height=6)
    accent_bar.pack(fill=X)

    Label(
        card,
        text=title,
        font=("Segoe UI", 10, "bold"),
        bg="#ffffff",
        fg="#576574",
        anchor="w",
    ).pack(fill=X, padx=14, pady=(14, 4))

    Label(
        card,
        textvariable=value_var,
        font=("Segoe UI", 20, "bold"),
        bg="#ffffff",
        fg="#1f2d3d",
        anchor="w",
    ).pack(fill=X, padx=14, pady=(0, 14))


def main_menu():
    ensure_setup()

    root = Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("820x620")
    root.minsize(780, 560)
    root.configure(bg="#f4efe6")

    stats_vars = {
        "users": StringVar(),
        "samples": StringVar(),
        "attendance": StringVar(),
        "status": StringVar(),
        "latest": StringVar(),
    }

    def refresh_dashboard(status_message="Ready to capture, train, or recognize."):
        stats_vars["users"].set(str(count_registered_users()))
        stats_vars["samples"].set(str(count_dataset_images()))
        stats_vars["attendance"].set(str(count_attendance_entries()))
        stats_vars["latest"].set(get_last_attendance_entry())
        stats_vars["status"].set(status_message)

    def run_task(task, success_message):
        success = task()
        if success:
            refresh_dashboard(success_message)
        else:
            refresh_dashboard("Action canceled or not completed.")

    hero = Frame(root, bg="#143642", padx=28, pady=24)
    hero.pack(fill=X, padx=20, pady=(20, 12))

    Label(
        hero,
        text="Face Recognition Attendance",
        font=("Segoe UI", 24, "bold"),
        bg="#143642",
        fg="#fefefe",
    ).pack(anchor="w")

    Label(
        hero,
        text="Register users, train the model, and record attendance from one dashboard.",
        font=("Segoe UI", 11),
        bg="#143642",
        fg="#d8f3dc",
    ).pack(anchor="w", pady=(8, 0))

    stats_frame = Frame(root, bg="#f4efe6")
    stats_frame.pack(fill=X, padx=20, pady=(0, 12))

    build_stat_card(stats_frame, "Registered Users", stats_vars["users"], "#ff9f43")
    build_stat_card(stats_frame, "Face Samples", stats_vars["samples"], "#20bf6b")
    build_stat_card(stats_frame, "Attendance Entries", stats_vars["attendance"], "#3867d6")

    content = Frame(root, bg="#f4efe6")
    content.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))

    actions = Frame(content, bg="#fffdf8", padx=20, pady=20)
    actions.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

    sidebar = Frame(content, bg="#fffdf8", padx=20, pady=20)
    sidebar.pack(side=LEFT, fill=BOTH, expand=True)

    Label(
        actions,
        text="Quick Actions",
        font=("Segoe UI", 16, "bold"),
        bg="#fffdf8",
        fg="#1f2d3d",
    ).pack(anchor="w")

    Label(
        actions,
        text="Use the buttons below to manage registration, training, and attendance.",
        font=("Segoe UI", 10),
        bg="#fffdf8",
        fg="#576574",
        wraplength=340,
        justify=LEFT,
    ).pack(anchor="w", pady=(8, 18))

    button_style = {
        "font": ("Segoe UI", 11, "bold"),
        "width": 24,
        "height": 2,
        "bd": 0,
        "cursor": "hand2",
    }

    Button(
        actions,
        text="Register Face",
        command=lambda: run_task(
            collect_data, "Face registration completed successfully."
        ),
        bg="#ff9f43",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Train Model",
        command=lambda: run_task(train_model, "Model trained and saved successfully."),
        bg="#20bf6b",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Recognize Face",
        command=lambda: run_task(
            recognize_faces, "Recognition session finished successfully."
        ),
        bg="#3867d6",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Preview Attendance",
        command=show_attendance_preview,
        bg="#5f27cd",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Open Project Folder",
        command=open_project_folder,
        bg="#576574",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Refresh Dashboard",
        command=lambda: refresh_dashboard("Dashboard refreshed."),
        bg="#0abde3",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=6)

    Button(
        actions,
        text="Exit",
        command=root.quit,
        bg="#ee5253",
        fg="white",
        **button_style,
    ).pack(anchor="w", pady=(6, 0))

    Label(
        sidebar,
        text="Project Snapshot",
        font=("Segoe UI", 16, "bold"),
        bg="#fffdf8",
        fg="#1f2d3d",
    ).pack(anchor="w")

    latest_card = Frame(sidebar, bg="#f7f1e3", padx=16, pady=16)
    latest_card.pack(fill=X, pady=(16, 12))

    Label(
        latest_card,
        text="Latest Attendance",
        font=("Segoe UI", 10, "bold"),
        bg="#f7f1e3",
        fg="#576574",
    ).pack(anchor="w")

    Label(
        latest_card,
        textvariable=stats_vars["latest"],
        font=("Segoe UI", 12, "bold"),
        bg="#f7f1e3",
        fg="#1f2d3d",
        wraplength=250,
        justify=LEFT,
    ).pack(anchor="w", pady=(8, 0))

    status_card = Frame(sidebar, bg="#eaf7ff", padx=16, pady=16)
    status_card.pack(fill=X, pady=(0, 12))

    Label(
        status_card,
        text="System Status",
        font=("Segoe UI", 10, "bold"),
        bg="#eaf7ff",
        fg="#576574",
    ).pack(anchor="w")

    Label(
        status_card,
        textvariable=stats_vars["status"],
        font=("Segoe UI", 11),
        bg="#eaf7ff",
        fg="#1f2d3d",
        wraplength=250,
        justify=LEFT,
    ).pack(anchor="w", pady=(8, 0))

    tips_card = Frame(sidebar, bg="#fff4d6", padx=16, pady=16)
    tips_card.pack(fill=BOTH, expand=True)

    Label(
        tips_card,
        text="Usage Tips",
        font=("Segoe UI", 10, "bold"),
        bg="#fff4d6",
        fg="#576574",
    ).pack(anchor="w")

    Label(
        tips_card,
        text=(
            "1. Register a new user.\n"
            "2. Train the model after adding samples.\n"
            "3. Start recognition to mark attendance.\n"
            "4. Use Preview Attendance to verify recent entries."
        ),
        font=("Segoe UI", 10),
        bg="#fff4d6",
        fg="#1f2d3d",
        justify=LEFT,
        wraplength=250,
    ).pack(anchor="w", pady=(8, 0))

    refresh_dashboard()
    root.mainloop()


if __name__ == "__main__":
    ensure_setup()
    main_menu()
