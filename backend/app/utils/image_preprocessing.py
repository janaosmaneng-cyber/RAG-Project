"""Preprocessing applied to user-uploaded images before they reach the
YOLO classifiers.

The classification datasets (Down syndrome, autism, depression/emotion)
are all tightly-cropped, upright, front-facing headshots. A raw phone or
webcam upload is none of these things by default: it may carry an EXIF
rotation tag the model never sees, and the face may only occupy a small,
off-center fraction of the frame. Both of these cause silent accuracy
collapse even though the underlying model is fine — this module closes
that gap before the image is ever handed to `model(...)`.
"""

import cv2
from PIL import Image, ImageOps

from app.utils.logging_config import logger

# Loaded once at import time; reused across every request.
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class NoFaceDetectedError(Exception):
    """Raised when no face can be found in an uploaded image. The caller
    (the /query route) should turn this into a user-facing 422 asking
    for a clearer photo, rather than silently classifying the full,
    unrelated frame."""


def normalize_orientation(image_path: str) -> None:
    """Bake in EXIF rotation so downstream libraries (cv2/YOLO, which
    ignore EXIF entirely) see the image the way a human would. Overwrites
    the file in place."""
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_path)
    except Exception:
        logger.exception(
            "EXIF normalization failed for %s; continuing with original file.",
            image_path,
        )


def crop_to_face(image_path: str, margin_ratio: float = 0.25) -> str:
    """Detect the largest face in the image and crop to it, with a margin.

    Returns a path to the cropped image (a sibling file), or the original
    path unchanged if no face is confidently detected. The Haar cascade
    used here is fast but not very robust (it can miss faces at an angle,
    mid-smile, or partly covered by hair), so treating "no detection" as
    a hard error blocks plenty of perfectly normal photos. Falling back
    to the full frame is the safer default; see NoFaceDetectedError below
    if you want to switch back to the strict behavior later.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image for face-cropping: %s", image_path)
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        logger.info("No face detected in %s; using full frame.", image_path)
        return image_path

    # Largest bounding box wins (most likely the primary subject).
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(img.shape[1], x + w + margin_x)
    y1 = min(img.shape[0], y + h + margin_y)

    cropped = img[y0:y1, x0:x1]

    root, ext = image_path.rsplit(".", 1) if "." in image_path else (image_path, "jpg")
    cropped_path = f"{root}_face.{ext}"
    cv2.imwrite(cropped_path, cropped)

    return cropped_path


def preprocess_for_classification(image_path: str) -> str:
    """Full pipeline: fix orientation in place, then crop to the detected
    face (or fall back to the full frame if no face is found). Returns
    the path that should actually be passed to YOLO."""
    normalize_orientation(image_path)
    return crop_to_face(image_path)