"""
Real-Time Emotion Detection using OpenCV + DeepFace
=====================================================
Compatible with Python 3.13+

Install:
    pip install opencv-python deepface tf-keras

Run:
    python emotion_detection.py

Press Q to quit.
"""

import cv2
import time
import threading
from deepface import DeepFace

# ── Emotion colour palette (BGR) ─────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":     (0,   0,   220),
    "disgust":   (0,   140,  0),
    "fear":      (180,  0,  180),
    "happy":     (0,   215, 255),
    "sad":       (220,  80,  20),
    "surprise":  (0,   200, 255),
    "neutral":   (160, 160, 160),
}

# ── Shared state for threaded detection ──────────────────────────────────────
latest_results = []
analysis_lock  = threading.Lock()
is_analyzing   = False


def analyze_frame(frame):
    """Run DeepFace in a background thread so the camera stays smooth."""
    global latest_results, is_analyzing
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if not isinstance(results, list):
            results = [results]
        with analysis_lock:
            latest_results = results
    except Exception:
        with analysis_lock:
            latest_results = []
    finally:
        is_analyzing = False


def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, thickness=2):
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius),   0, 0, 90, color, thickness)


def draw_emotion_bar(frame, emotion, score, y_pos, bar_x=15, bar_max_w=160):
    color  = EMOTION_COLORS.get(emotion, (200, 200, 200))
    filled = int(bar_max_w * min(score / 100.0, 1.0))
    label  = f"{emotion:<9} {score:5.1f}%"
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x+bar_max_w, y_pos+14), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x+filled, y_pos+14), color, -1)
    cv2.putText(frame, label, (bar_x+bar_max_w+6, y_pos+11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def overlay_hud(frame, fps, analyzing):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
    status = "Analyzing..." if analyzing else "Live"
    cv2.putText(frame, status, (w-110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (w-150, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


def run():
    global is_analyzing

    print("Loading DeepFace models — please wait a few seconds on first run...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Make sure no other app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened. Detecting emotions — press Q to quit.")

    DETECT_EVERY = 5
    frame_idx    = 0
    t_prev       = time.time()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — exiting.")
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]

        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        if frame_idx % DETECT_EVERY == 0 and not is_analyzing:
            is_analyzing = True
            t = threading.Thread(target=analyze_frame, args=(frame.copy(),), daemon=True)
            t.start()
        frame_idx += 1

        with analysis_lock:
            results_snapshot = list(latest_results)

        for face_data in results_snapshot:
            emotions = face_data.get("emotion", {})
            region   = face_data.get("region", {})

            x  = region.get("x", 0)
            y  = region.get("y", 0)
            fw = region.get("w", 0)
            fh = region.get("h", 0)

            if fw == 0 or fh == 0:
                continue

            dominant = face_data.get("dominant_emotion", max(emotions, key=emotions.get, default="neutral"))
            score    = emotions.get(dominant, 0)
            color    = EMOTION_COLORS.get(dominant, (200, 200, 200))

            draw_rounded_rect(frame, x, y, x+fw, y+fh, color, radius=14, thickness=2)

            label = f"{dominant.upper()}  {score:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            pill_x1 = x
            pill_y1 = max(y - th - 16, 0)
            pill_x2 = x + tw + 16
            pill_y2 = max(y - 4, th + 12)

            overlay = frame.copy()
            cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, label, (pill_x1+8, pill_y2-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            sorted_emotions = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)
            n = len(sorted_emotions)
            panel_x       = 10
            panel_y_start = h - n * 22 - 15
            panel_w       = 295
            panel_h       = n * 22 + 10

            bg = frame.copy()
            cv2.rectangle(bg, (panel_x-5, panel_y_start-5),
                          (panel_x+panel_w, panel_y_start+panel_h), (20, 20, 20), -1)
            cv2.addWeighted(bg, 0.55, frame, 0.45, 0, frame)

            for i, (emo, sc) in enumerate(sorted_emotions):
                draw_emotion_bar(frame, emo, sc, panel_y_start + i*22, bar_x=panel_x+5)

        if not results_snapshot:
            msg = "No face detected — look at the camera"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(frame, msg, ((w-tw)//2, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 1, cv2.LINE_AA)

        overlay_hud(frame, fps, is_analyzing)

        cv2.imshow("Emotion Detection  |  press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    run()



















"""
Real-Time Emotion Detection — Cyberpunk HUD Edition
=====================================================
Compatible with Python 3.13+

Install:
    pip install opencv-python deepface tf-keras numpy

Run:
    python emotion_detection.py

Controls:
    Q        — Quit
    S        — Save screenshot
    +/-      — Increase / decrease detection sensitivity
"""

# import cv2
# import time
# import threading
# import numpy as np
# from collections import deque
# from deepface import DeepFace
# import os
# from datetime import datetime

# # ══════════════════════════════════════════════════════════════════════════════
# #  CONFIG
# # ══════════════════════════════════════════════════════════════════════════════
# CAM_W, CAM_H   = 1280, 720
# DETECT_EVERY   = 4          # analyze every N frames
# SMOOTH_WINDOW  = 6          # frames to average for stable readings
# SCREENSHOTS_DIR = "screenshots"

# # ── Cyberpunk colour palette (BGR) ───────────────────────────────────────────
# NEON = {
#     "cyan":    (255, 230,  0),
#     "magenta": (180,   0, 255),
#     "yellow":  (  0, 230, 255),
#     "green":   (  0, 255, 140),
#     "red":     (  0,  30, 255),
#     "blue":    (255, 100,  30),
#     "white":   (220, 220, 220),
#     "dim":     ( 80,  80,  80),
#     "bg":      ( 10,  10,  18),
# }

# EMOTION_NEON = {
#     "happy":    NEON["yellow"],
#     "sad":      NEON["blue"],
#     "angry":    NEON["red"],
#     "fear":     NEON["magenta"],
#     "surprise": NEON["cyan"],
#     "disgust":  NEON["green"],
#     "neutral":  NEON["white"],
# }

# EMOTION_ICON = {
#     "happy":    "HAPPY",
#     "sad":      "SAD",
#     "angry":    "ANGRY",
#     "fear":     "FEAR",
#     "surprise": "SURPR",
#     "disgust":  "DISGST",
#     "neutral":  "NEUTRAL",
# }

# # ══════════════════════════════════════════════════════════════════════════════
# #  SHARED STATE
# # ══════════════════════════════════════════════════════════════════════════════
# latest_results  = []
# analysis_lock   = threading.Lock()
# is_analyzing    = False
# emotion_history = deque(maxlen=SMOOTH_WINDOW)   # list of emotion-dicts


# # ══════════════════════════════════════════════════════════════════════════════
# #  DEEPFACE THREAD
# # ══════════════════════════════════════════════════════════════════════════════
# def analyze_frame(frame):
#     global latest_results, is_analyzing
#     try:
#         small = cv2.resize(frame, (640, 360))
#         sx, sy = frame.shape[1] / 640, frame.shape[0] / 360
#         results = DeepFace.analyze(
#             small,
#             actions=["emotion"],
#             enforce_detection=False,
#             detector_backend="opencv",
#             silent=True,
#         )
#         if not isinstance(results, list):
#             results = [results]
#         # scale regions back to full resolution
#         for r in results:
#             reg = r.get("region", {})
#             reg["x"] = int(reg.get("x", 0) * sx)
#             reg["y"] = int(reg.get("y", 0) * sy)
#             reg["w"] = int(reg.get("w", 0) * sx)
#             reg["h"] = int(reg.get("h", 0) * sy)
#         with analysis_lock:
#             latest_results = results
#             emotion_history.append(results[0]["emotion"] if results else {})
#     except Exception:
#         with analysis_lock:
#             latest_results = []
#     finally:
#         is_analyzing = False


# # ══════════════════════════════════════════════════════════════════════════════
# #  DRAWING HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def blend_rect(img, x1, y1, x2, y2, color, alpha=0.35):
#     """Semi-transparent filled rectangle."""
#     sub = img[y1:y2, x1:x2]
#     if sub.size == 0:
#         return
#     overlay = np.full_like(sub, color)
#     cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
#     img[y1:y2, x1:x2] = sub


# def corner_bracket(img, x1, y1, x2, y2, color, size=22, thick=2):
#     """Draw only the 4 corner brackets of a rectangle — HUD style."""
#     pts = [
#         # top-left
#         ((x1, y1 + size), (x1, y1), (x1 + size, y1)),
#         # top-right
#         ((x2 - size, y1), (x2, y1), (x2, y1 + size)),
#         # bottom-left
#         ((x1, y2 - size), (x1, y2), (x1 + size, y2)),
#         # bottom-right
#         ((x2 - size, y2), (x2, y2), (x2, y2 - size)),
#     ]
#     for p in pts:
#         cv2.polylines(img, [np.array(p)], False, color, thick, cv2.LINE_AA)


# def scanline_overlay(img, alpha=0.04):
#     """Subtle scanline effect every 3 pixels."""
#     h, w = img.shape[:2]
#     for y in range(0, h, 3):
#         img[y, :] = (img[y, :] * (1 - alpha)).astype(np.uint8)


# def glow_text(img, text, pos, font, scale, color, thick=1, glow_radius=2):
#     """Text with a soft glow halo."""
#     # glow (darker, spread)
#     glow_color = tuple(int(c * 0.4) for c in color)
#     for dx in range(-glow_radius, glow_radius + 1):
#         for dy in range(-glow_radius, glow_radius + 1):
#             if dx == 0 and dy == 0:
#                 continue
#             cv2.putText(img, text, (pos[0]+dx, pos[1]+dy),
#                         font, scale, glow_color, thick + 1, cv2.LINE_AA)
#     cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


# def draw_hex_badge(img, cx, cy, radius, color, label, score):
#     """Hexagonal emotion badge."""
#     pts = []
#     for i in range(6):
#         angle = np.radians(60 * i - 30)
#         pts.append((int(cx + radius * np.cos(angle)),
#                     int(cy + radius * np.sin(angle))))
#     pts_arr = np.array(pts)
#     blend_rect(img,
#                max(cx - radius, 0), max(cy - radius, 0),
#                min(cx + radius, img.shape[1]), min(cy + radius, img.shape[0]),
#                color, alpha=0.15)
#     cv2.polylines(img, [pts_arr], True, color, 2, cv2.LINE_AA)
#     glow_text(img, label, (cx - 28, cy - 6),
#               cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
#     glow_text(img, f"{score:.0f}%", (cx - 18, cy + 14),
#               cv2.FONT_HERSHEY_SIMPLEX, 0.48, NEON["white"], 1)


# def draw_emotion_radar(img, emotions, cx, cy, r=90):
#     """Radar / spider chart of all 7 emotions."""
#     keys   = list(emotions.keys())
#     n      = len(keys)
#     angles = [np.radians(360 / n * i - 90) for i in range(n)]

#     # grid rings
#     for ring in [0.25, 0.5, 0.75, 1.0]:
#         ring_pts = [(int(cx + r * ring * np.cos(a)),
#                      int(cy + r * ring * np.sin(a))) for a in angles]
#         cv2.polylines(img, [np.array(ring_pts)], True,
#                       (40, 40, 60), 1, cv2.LINE_AA)

#     # spokes
#     for a in angles:
#         cv2.line(img, (cx, cy),
#                  (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))),
#                  (40, 40, 60), 1, cv2.LINE_AA)

#     # data polygon
#     data_pts = []
#     for i, key in enumerate(keys):
#         val = emotions[key] / 100.0
#         px  = int(cx + r * val * np.cos(angles[i]))
#         py  = int(cy + r * val * np.sin(angles[i]))
#         data_pts.append((px, py))

#     dominant = max(emotions, key=emotions.get)
#     fill_color = EMOTION_NEON.get(dominant, NEON["cyan"])

#     # filled polygon (semi-transparent)
#     poly_overlay = img.copy()
#     cv2.fillPoly(poly_overlay, [np.array(data_pts)], fill_color)
#     cv2.addWeighted(poly_overlay, 0.25, img, 0.75, 0, img)
#     cv2.polylines(img, [np.array(data_pts)], True, fill_color, 2, cv2.LINE_AA)

#     # labels
#     for i, key in enumerate(keys):
#         lx = int(cx + (r + 18) * np.cos(angles[i]))
#         ly = int(cy + (r + 18) * np.sin(angles[i]))
#         short = key[:3].upper()
#         cv2.putText(img, short, (lx - 12, ly + 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.32,
#                     EMOTION_NEON.get(key, NEON["dim"]), 1, cv2.LINE_AA)


# def draw_bar_panel(img, emotions, x, y, w=220, bar_h=12, gap=8):
#     """Vertical stack of slim labelled bars."""
#     sorted_emo = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)
#     panel_h    = len(sorted_emo) * (bar_h + gap) + 10
#     blend_rect(img, x - 8, y - 8, x + w + 60, y + panel_h, (10, 10, 20), alpha=0.7)
#     cv2.rectangle(img, (x - 8, y - 8), (x + w + 60, y + panel_h),
#                   (40, 40, 60), 1)

#     for i, (emo, score) in enumerate(sorted_emo):
#         color   = EMOTION_NEON.get(emo, NEON["white"])
#         by      = y + i * (bar_h + gap)
#         filled  = int(w * score / 100.0)

#         # track
#         cv2.rectangle(img, (x, by), (x + w, by + bar_h), (30, 30, 40), -1)
#         # fill
#         if filled > 0:
#             cv2.rectangle(img, (x, by), (x + filled, by + bar_h), color, -1)
#             # bright leading edge
#             cv2.line(img, (x + filled, by), (x + filled, by + bar_h),
#                      NEON["white"], 1)

#         label = f"{emo.upper():<8}  {score:5.1f}%"
#         cv2.putText(img, label, (x + w + 4, by + bar_h - 1),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)


# def draw_top_bar(img, fps, analyzing, frame_count):
#     """Top HUD strip."""
#     h, w = img.shape[:2]
#     blend_rect(img, 0, 0, w, 36, (10, 10, 20), alpha=0.75)
#     cv2.line(img, (0, 36), (w, 36), NEON["cyan"], 1)

#     # Left — title
#     glow_text(img, "[ EMOTION SCANNER v2.0 ]", (12, 24),
#               cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON["cyan"], 1)

#     # Centre — frame counter
#     fc_txt = f"FRAME : {frame_count:06d}"
#     (tw, _), _ = cv2.getTextSize(fc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
#     cv2.putText(img, fc_txt, ((w - tw) // 2, 24),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON["dim"], 1, cv2.LINE_AA)

#     # Right — FPS + status
#     status_color = NEON["yellow"] if analyzing else NEON["green"]
#     status_txt   = "ANALYZING" if analyzing else "LIVE"
#     glow_text(img, f"FPS {fps:5.1f}   {status_txt}", (w - 220, 24),
#               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)


# def draw_bottom_bar(img):
#     h, w = img.shape[:2]
#     blend_rect(img, 0, h - 28, w, h, (10, 10, 20), alpha=0.75)
#     cv2.line(img, (0, h - 28), (w, h - 28), NEON["cyan"], 1)
#     hints = "[ Q ] QUIT     [ S ] SCREENSHOT"
#     (tw, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
#     cv2.putText(img, hints, ((w - tw) // 2, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON["dim"], 1, cv2.LINE_AA)


# def smooth_emotions(history):
#     """Average emotion scores over the history window."""
#     if not history:
#         return {}
#     keys = history[-1].keys()
#     avg  = {}
#     for k in keys:
#         vals = [h.get(k, 0) for h in history if h]
#         avg[k] = sum(vals) / len(vals) if vals else 0
#     return avg


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN LOOP
# # ══════════════════════════════════════════════════════════════════════════════
# def run():
#     global is_analyzing

#     os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
#     print("Loading DeepFace — first run downloads models, please wait...")

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
#     print("Camera ready. Press Q to quit, S to screenshot.")

#     frame_idx = 0
#     t_prev    = time.time()
#     fps       = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         h, w  = frame.shape[:2]

#         # FPS
#         t_now  = time.time()
#         fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
#         t_prev = t_now

#         # Kick off background analysis
#         if frame_idx % DETECT_EVERY == 0 and not is_analyzing:
#             is_analyzing = True
#             threading.Thread(target=analyze_frame,
#                              args=(frame.copy(),), daemon=True).start()
#         frame_idx += 1

#         with analysis_lock:
#             results_snap = list(latest_results)
#             history_snap = list(emotion_history)

#         # ── Dark vignette background tint ────────────────────────────────────
#         vignette = np.zeros_like(frame, dtype=np.float32)
#         cv2.circle(vignette, (w // 2, h // 2),
#                    int(min(w, h) * 0.7), (1, 1, 1), -1)
#         vignette = cv2.GaussianBlur(vignette, (0, 0), sigmaX=w // 4)
#         frame    = (frame.astype(np.float32) * np.clip(vignette + 0.55, 0.55, 1.0)).astype(np.uint8)

#         # ── Scanlines ─────────────────────────────────────────────────────────
#         scanline_overlay(frame, alpha=0.03)

#         # ── Per-face rendering ────────────────────────────────────────────────
#         for face_data in results_snap:
#             emotions = face_data.get("emotion", {})
#             region   = face_data.get("region", {})
#             x, y     = region.get("x", 0), region.get("y", 0)
#             fw, fh   = region.get("w", 0), region.get("h", 0)
#             if fw == 0 or fh == 0:
#                 continue

#             smoothed = smooth_emotions(history_snap) if history_snap else emotions
#             dominant = max(smoothed, key=smoothed.get, default="neutral")
#             score    = smoothed.get(dominant, 0)
#             color    = EMOTION_NEON.get(dominant, NEON["cyan"])

#             # Face bounding — corner brackets only (HUD style)
#             pad = 10
#             corner_bracket(frame, x - pad, y - pad,
#                            x + fw + pad, y + fh + pad, color, size=28, thick=2)

#             # Thin full border (dim)
#             cv2.rectangle(frame, (x - pad, y - pad),
#                           (x + fw + pad, y + fh + pad),
#                           tuple(int(c * 0.25) for c in color), 1)

#             # Dominant emotion label — above face
#             badge_txt = f"[ {EMOTION_ICON.get(dominant, dominant.upper())} ]"
#             (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
#             bx = x + (fw - tw) // 2
#             by = max(y - pad - 10, th + 4)
#             blend_rect(frame, bx - 8, by - th - 4, bx + tw + 8, by + 4, color, alpha=0.2)
#             cv2.rectangle(frame, (bx - 8, by - th - 4), (bx + tw + 8, by + 4), color, 1)
#             glow_text(frame, badge_txt, (bx, by),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, glow_radius=3)

#             # Score below badge
#             score_txt = f"{score:.1f}%  CONFIDENCE"
#             (sw, _), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
#             cv2.putText(frame, score_txt, (bx + (tw - sw) // 2, by + 16),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON["white"], 1, cv2.LINE_AA)

#             # Crosshair on face centre
#             fcx, fcy = x + fw // 2, y + fh // 2
#             cv2.line(frame, (fcx - 12, fcy), (fcx + 12, fcy), color, 1, cv2.LINE_AA)
#             cv2.line(frame, (fcx, fcy - 12), (fcx, fcy + 12), color, 1, cv2.LINE_AA)
#             cv2.circle(frame, (fcx, fcy), 4, color, 1, cv2.LINE_AA)

#             # ── RIGHT PANEL: emotion bars ─────────────────────────────────────
#             panel_x = w - 295
#             panel_y = 50
#             draw_bar_panel(frame, smoothed, panel_x, panel_y, w=190)

#             # ── LEFT PANEL: radar chart ───────────────────────────────────────
#             radar_cx, radar_cy = 130, h - 160
#             blend_rect(frame,
#                        radar_cx - 118, radar_cy - 118,
#                        radar_cx + 118, radar_cy + 118,
#                        (10, 10, 20), alpha=0.65)
#             cv2.circle(frame, (radar_cx, radar_cy), 110, (30, 30, 50), 1)
#             draw_emotion_radar(frame, smoothed, radar_cx, radar_cy, r=90)
#             cv2.putText(frame, "EMOTION RADAR", (radar_cx - 56, radar_cy + 112),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.32, NEON["dim"], 1, cv2.LINE_AA)

#         # No face
#         if not results_snap:
#             msg = "NO FACE DETECTED"
#             (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#             glow_text(frame, msg, ((w - tw) // 2, h // 2),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON["red"], 2, glow_radius=4)
#             # scanning line animation
#             scan_y = int((time.time() * 180) % h)
#             cv2.line(frame, (0, scan_y), (w, scan_y),
#                      (*NEON["cyan"][:2], 60), 1, cv2.LINE_AA)

#         # ── HUD bars ──────────────────────────────────────────────────────────
#         draw_top_bar(frame, fps, is_analyzing, frame_idx)
#         draw_bottom_bar(frame)

#         cv2.imshow("EMOTION SCANNER  |  Q = quit  S = screenshot", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord("s"):
#             ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
#             path = os.path.join(SCREENSHOTS_DIR, f"emotion_{ts}.png")
#             cv2.imwrite(path, frame)
#             print(f"Screenshot saved: {path}")

#     cap.release()
#     cv2.destroyAllWindows()
#     print("Closed.")


# if __name__ == "__main__":
#     run()
