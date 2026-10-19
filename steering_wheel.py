import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
from pynput.keyboard import Key, Controller

CAMERA_INDEX       = 0
DEAD_ZONE_DEG      = 4.5
RELEASE_ZONE_DEG   = 2.0
SOFT_ZONE_DEG      = 20.0
FLIP_CAMERA        = True
SHOW_ANGLE         = True
MIN_DETECTION_CONF = 0.55
MIN_TRACKING_CONF  = 0.45
GRACE_FRAMES       = 4
CALIBRATION_FRAMES = 35
OPEN_FINGER_THRESH = 3

# Minimal, high-contrast dashboard palette (OpenCV uses BGR).
CLR_WHEEL   = (190, 220, 255)
CLR_LEFT    = (230, 150, 95)
CLR_RIGHT   = (115, 220, 160)
CLR_NEUTRAL = (185, 195, 210)
CLR_TEXT    = (245, 247, 250)
CLR_ACCENT  = (220, 185, 95)
CLR_HAND_L  = (230, 150, 95)
CLR_HAND_R  = (115, 220, 160)
CLR_ACCEL   = (115, 220, 160)
CLR_BRAKE   = (230, 150, 95)
CLR_MUTED   = (135, 145, 160)
CLR_PANEL   = (18, 22, 30)
CLR_TRACK   = (48, 56, 69)
CLR_BORDER  = (58, 68, 82)

keyboard   = Controller()
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]

    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if hand_landmarks.landmark[tip].y
        < hand_landmarks.landmark[pip].y
    )

    return extended >= OPEN_FINGER_THRESH


class SteeringController:
    def __init__(self):
        self.keys_held = {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }

        self.angle_history = []
        self.HISTORY_LEN = 1
        self.filtered_angle = None
        self.last_raw_angle = None
        self.last_angle_time = None
        self.neutral_angle = None
        self.calibration_samples = []

    @property
    def calibrated(self):
        return self.neutral_angle is not None

    @staticmethod
    def raw_hand_angle(left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        return math.degrees(math.atan2(dy, dx))

    def calibrate(self, left_wrist, right_wrist):
        """Learn the natural neutral wrist-line angle."""

        raw = self.raw_hand_angle(left_wrist, right_wrist)
        self.calibration_samples.append(raw)

        if len(self.calibration_samples) >= CALIBRATION_FRAMES:
            self.neutral_angle = float(
                np.median(self.calibration_samples)
            )

            self.filtered_angle = 0.0
            self.last_raw_angle = 0.0
            self.last_angle_time = time.perf_counter()

            print(
                "[INFO] Steering calibrated. Neutral angle: "
                f"{self.neutral_angle:.1f} degrees"
            )

        return len(self.calibration_samples), CALIBRATION_FRAMES

    def _press(self, key):
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if self.keys_held[key]:
            keyboard.release(key)
            self.keys_held[key] = False

    def release_all(self):
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass

            self.keys_held[key] = False

        self.angle_history.clear()
        self.filtered_angle = None
        self.last_raw_angle = None
        self.last_angle_time = None

    def smooth_angle(self, raw_angle):
        """Reduce tiny jitter while preserving quick steering changes."""

        now = time.perf_counter()

        if (
            self.filtered_angle is None
            or self.last_angle_time is None
        ):
            self.filtered_angle = raw_angle

        else:
            dt = max(now - self.last_angle_time, 1e-4)

            previous_raw = (
                self.last_raw_angle
                if self.last_raw_angle is not None
                else raw_angle
            )

            angular_speed = abs(raw_angle - previous_raw) / dt

            # Adaptive smoothing:
            # Fast turns pass through almost immediately.
            if angular_speed >= 100:
                alpha = 0.98
            elif angular_speed >= 45:
                alpha = 0.94
            elif angular_speed >= 18:
                alpha = 0.86
            else:
                alpha = 0.72

            self.filtered_angle += alpha * (
                raw_angle - self.filtered_angle
            )

        self.last_raw_angle = raw_angle
        self.last_angle_time = now

        return float(self.filtered_angle)

    def update_steer(self, left_wrist, right_wrist):
        raw_angle_deg = self.raw_hand_angle(
            left_wrist,
            right_wrist
        )

        # Do not steer until calibration has finished.
        if self.neutral_angle is None:
            self.calibrate(left_wrist, right_wrist)
            return 0.0, "STRAIGHT", 0.0

        # Calculate angle relative to the calibrated neutral position.
        raw_angle_deg = (
            (raw_angle_deg - self.neutral_angle + 180.0)
            % 360.0
        ) - 180.0

        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"

        if angle < -DEAD_ZONE_DEG:
            direction = "LEFT"

        elif angle > DEAD_ZONE_DEG:
            direction = "RIGHT"

        elif (
            self.keys_held["a"]
            and angle > -RELEASE_ZONE_DEG
        ):
            direction = "STRAIGHT"

        elif (
            self.keys_held["d"]
            and angle < RELEASE_ZONE_DEG
        ):
            direction = "STRAIGHT"

        strength = 0.0

        if direction == "LEFT":
            strength = min(
                1.0,
                (abs(angle) - DEAD_ZONE_DEG)
                / (SOFT_ZONE_DEG - DEAD_ZONE_DEG)
            )

            # Release the opposite key before switching direction.
            self._release("d")
            self._press("a")

        elif direction == "RIGHT":
            strength = min(
                1.0,
                (abs(angle) - DEAD_ZONE_DEG)
                / (SOFT_ZONE_DEG - DEAD_ZONE_DEG)
            )

            self._release("a")
            self._press("d")

        else:
            self._release("a")
            self._release("d")

        return angle, direction, strength

    def update_throttle(self, left_open, right_open):
        # RIGHT FIST = W (accelerate).
        # RIGHT OPEN PALM = S (brake).
        if not right_open:
            self._release("s")
            self._press("w")
            return "ACCEL"

        self._release("w")
        self._press("s")
        return "BRAKE"


def rounded_rect(
    frame,
    x1,
    y1,
    x2,
    y2,
    color,
    radius=14,
    thickness=-1
):
    """Draw a clean rounded rectangle using OpenCV primitives."""

    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return

    r = max(
        1,
        min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    )

    if thickness < 0:
        cv2.rectangle(
            frame,
            (x1 + r, y1),
            (x2 - r, y2),
            color,
            -1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            frame,
            (x1, y1 + r),
            (x2, y2 - r),
            color,
            -1,
            cv2.LINE_AA
        )

        for center in (
            (x1 + r, y1 + r),
            (x2 - r, y1 + r),
            (x1 + r, y2 - r),
            (x2 - r, y2 - r),
        ):
            cv2.circle(
                frame, center, r, color, -1, cv2.LINE_AA
            )

    else:
        cv2.line(
            frame,
            (x1 + r, y1),
            (x2 - r, y1),
            color,
            thickness,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (x1 + r, y2),
            (x2 - r, y2),
            color,
            thickness,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (x1, y1 + r),
            (x1, y2 - r),
            color,
            thickness,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (x2, y1 + r),
            (x2, y2 - r),
            color,
            thickness,
            cv2.LINE_AA
        )

        for center, a1, a2 in (
            ((x1 + r, y1 + r), 180, 270),
            ((x2 - r, y1 + r), 270, 360),
            ((x2 - r, y2 - r), 0, 90),
            ((x1 + r, y2 - r), 90, 180),
        ):
            cv2.ellipse(
                frame,
                center,
                (r, r),
                0,
                a1,
                a2,
                color,
                thickness,
                cv2.LINE_AA
            )


def draw_steering_wheel(
    frame,
    center,
    angle_deg,
    direction,
    strength
):
    h, w = frame.shape[:2]

    radius = max(24, int(min(w, h) * 0.075))
    cx, cy = center

    color = CLR_NEUTRAL

    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(
        frame,
        (cx, cy),
        radius + 5,
        (32, 38, 48),
        5,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        CLR_TRACK,
        4,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        color,
        2,
        cv2.LINE_AA
    )

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)

        x1 = int(cx + radius * 0.22 * math.cos(rad))
        y1 = int(cy - radius * 0.22 * math.sin(rad))
        x2 = int(cx + radius * 0.83 * math.cos(rad))
        y2 = int(cy - radius * 0.83 * math.sin(rad))

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3,
            cv2.LINE_AA
        )

    cv2.circle(
        frame,
        (cx, cy),
        max(4, radius // 8),
        CLR_TEXT,
        -1,
        cv2.LINE_AA
    )

    if direction != "STRAIGHT":
        cv2.ellipse(
            frame,
            (cx, cy),
            (radius + 7, radius + 7),
            0,
            -35 if direction == "RIGHT" else 145,
            35 if direction == "RIGHT" else 215,
            color,
            3,
            cv2.LINE_AA
        )


def draw_hud(
    frame,
    angle,
    direction,
    strength,
    throttle_mode,
    both_hands_visible,
    left_open,
    right_open,
    fps,
    calibrated=True
):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.58, w / 1150))
    label_scale = max(0.34, scale * 0.72)
    value_scale = max(0.55, scale * 1.12)

    # Header.
    overlay = frame.copy()

    rounded_rect(
        overlay,
        12,
        12,
        w - 12,
        62,
        CLR_PANEL,
        16
    )

    cv2.addWeighted(
        overlay, 0.86, frame, 0.14, 0, frame
    )

    cv2.circle(
        frame,
        (31, 37),
        5,
        CLR_RIGHT if both_hands_visible else CLR_LEFT,
        -1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "VIRTUAL DRIVE",
        (45, 42),
        font,
        scale,
        CLR_TEXT,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "HAND TRACKING  /  LIVE",
        (int(w * 0.40), 40),
        font,
        max(0.34, scale * 0.72),
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    if calibrated:
        status = (
            "TRACKING ACTIVE"
            if both_hands_visible
            else "SHOW BOTH HANDS"
        )
    else:
        status = "CALIBRATING - HOLD HANDS STRAIGHT"

    if both_hands_visible and calibrated:
        status_color = CLR_RIGHT
    elif not calibrated:
        status_color = CLR_ACCENT
    else:
        status_color = CLR_LEFT

    status_size = max(0.34, scale * 0.72)

    status_width = cv2.getTextSize(
        status, font, status_size, 1
    )[0][0]

    cv2.putText(
        frame,
        status,
        (w - 28 - status_width, 40),
        font,
        status_size,
        status_color,
        1,
        cv2.LINE_AA
    )

    # Bottom dashboard panel.
    panel_top = max(0, h - 142)
    overlay = frame.copy()

    rounded_rect(
        overlay,
        12,
        panel_top,
        w - 12,
        h - 12,
        CLR_PANEL,
        16
    )

    cv2.addWeighted(
        overlay, 0.91, frame, 0.09, 0, frame
    )

    left_pad = 28
    right_pad = w - 28

    cv2.putText(
        frame,
        "STEERING",
        (left_pad, panel_top + 24),
        font,
        label_scale,
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    dir_color = (
        CLR_LEFT if direction == "LEFT"
        else CLR_RIGHT if direction == "RIGHT"
        else CLR_TEXT
    )

    cv2.putText(
        frame,
        direction,
        (left_pad, panel_top + 51),
        font,
        value_scale,
        dir_color,
        2,
        cv2.LINE_AA
    )

    angle_text = f"{angle:+.1f} deg" if SHOW_ANGLE else ""

    cv2.putText(
        frame,
        angle_text,
        (left_pad, panel_top + 75),
        font,
        label_scale,
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    # Steering strength meter.
    meter_x1 = int(w * 0.30)
    meter_x2 = int(w * 0.76)
    meter_y = panel_top + 48
    meter_h = 10

    rounded_rect(
        frame,
        meter_x1,
        meter_y,
        meter_x2,
        meter_y + meter_h,
        CLR_TRACK,
        5
    )

    mid = (meter_x1 + meter_x2) // 2

    cv2.line(
        frame,
        (mid, meter_y - 5),
        (mid, meter_y + meter_h + 5),
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    fill_len = int(
        ((meter_x2 - meter_x1) // 2)
        * max(0.0, min(1.0, strength))
    )

    if fill_len > 0 and direction == "LEFT":
        rounded_rect(
            frame,
            mid - fill_len,
            meter_y,
            mid,
            meter_y + meter_h,
            CLR_LEFT,
            5
        )

    elif fill_len > 0 and direction == "RIGHT":
        rounded_rect(
            frame,
            mid,
            meter_y,
            mid + fill_len,
            meter_y + meter_h,
            CLR_RIGHT,
            5
        )

    cv2.putText(
        frame,
        "LEFT",
        (meter_x1, meter_y - 10),
        font,
        label_scale * 0.85,
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    right_text = "RIGHT"

    right_width = cv2.getTextSize(
        right_text, font, label_scale * 0.85, 1
    )[0][0]

    cv2.putText(
        frame,
        right_text,
        (meter_x2 - right_width, meter_y - 10),
        font,
        label_scale * 0.85,
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    # Throttle pill.
    throttle_color = (
        CLR_ACCEL if throttle_mode == "ACCEL"
        else CLR_BRAKE if throttle_mode == "BRAKE"
        else CLR_NEUTRAL
    )

    throttle_label = {
        "ACCEL": "ACCELERATE  W",
        "BRAKE": "BRAKE  S",
        "NEUTRAL": "COAST",
    }[throttle_mode]

    pill_x1 = int(w * 0.67)
    pill_x2 = w - 28
    pill_y1 = panel_top + 78
    pill_y2 = panel_top + 112

    rounded_rect(
        frame,
        pill_x1,
        pill_y1,
        pill_x2,
        pill_y2,
        (35, 42, 53),
        10
    )

    rounded_rect(
        frame,
        pill_x1,
        pill_y1,
        pill_x1 + 4,
        pill_y2,
        throttle_color,
        2
    )

    cv2.putText(
        frame,
        throttle_label,
        (pill_x1 + 14, pill_y1 + 22),
        font,
        max(0.37, scale * 0.77),
        throttle_color,
        1,
        cv2.LINE_AA
    )

    l_label = "OPEN" if left_open else "FIST"
    r_label = "OPEN" if right_open else "FIST"

    hand_text = f"L {l_label}    R {r_label}"

    cv2.putText(
        frame,
        hand_text,
        (left_pad, h - 27),
        font,
        label_scale,
        CLR_TEXT,
        1,
        cv2.LINE_AA
    )

    fps_text = f"{fps:.0f} FPS"

    fps_width = cv2.getTextSize(
        fps_text, font, label_scale, 1
    )[0][0]

    cv2.putText(
        frame,
        fps_text,
        (right_pad - fps_width, h - 27),
        font,
        label_scale,
        CLR_MUTED,
        1,
        cv2.LINE_AA
    )

    # Compact wheel preview.
    draw_steering_wheel(
        frame,
        (w - 72, max(104, panel_top - 54)),
        angle,
        direction,
        strength
    )


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw

    cv2.line(
        frame,
        (lx, ly),
        (rx, ry),
        (24, 31, 42),
        9,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (lx, ly),
        (rx, ry),
        CLR_ACCENT,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame, (lx, ly), 9, CLR_HAND_L, -1, cv2.LINE_AA
    )

    cv2.circle(
        frame, (rx, ry), 9, CLR_HAND_R, -1, cv2.LINE_AA
    )

    cv2.circle(
        frame, (lx, ly), 13, CLR_TEXT, 1, cv2.LINE_AA
    )

    cv2.circle(
        frame, (rx, ry), 13, CLR_TEXT, 1, cv2.LINE_AA
    )

    mx = (lx + rx) // 2
    my = (ly + ry) // 2

    cv2.circle(
        frame, (mx, my), 7, CLR_WHEEL, -1
    )


def main():
    # Try suitable camera backends and common camera indexes.
    system = platform.system()

    if system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    elif system == "Windows":
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY,
        ]

    else:
        backends = [cv2.CAP_ANY]

    indexes = list(dict.fromkeys([CAMERA_INDEX, 0, 1, 2]))

    cap = None
    active_index = None

    for index in indexes:
        for backend in backends:
            candidate = cv2.VideoCapture(index, backend)

            if candidate.isOpened():
                ok, _ = candidate.read()

                if ok:
                    cap = candidate
                    active_index = index
                    break

            candidate.release()

        if cap is not None:
            break

    if cap is None:
        print("[ERROR] Cannot open a working camera.")
        print(
            "  -> Close other apps using the webcam and check "
            "Windows Camera permissions."
        )
        print(
            "  -> Camera indexes 0, 1, and 2 were tried."
        )
        return

    print(
        f"[INFO] Camera opened successfully (index {active_index})."
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Keep camera latency low where supported.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass

    controller = SteeringController()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )

    conn_style = mp_drawing.DrawingSpec(
        color=(80, 80, 100),
        thickness=1
    )

    landmark_style = mp_drawing.DrawingSpec(
        color=(200, 200, 255),
        thickness=1,
        circle_radius=2
    )

    prev_time = time.time()

    angle = 0.0
    direction = "STRAIGHT"
    strength = 0.0
    throttle_mode = "NEUTRAL"

    left_open = False
    right_open = False
    lost_frames = 0

    print("=" * 55)
    print("  Virtual Steering Wheel  |  Press Q to quit")
    print("=" * 55)
    print("  RIGHT FIST = W (accelerate)")
    print("  RIGHT OPEN PALM = S (brake)")
    print("  A = left steering, D = right steering")
    print("  Hold both hands straight for initial calibration.")
    print("  Press C to recalibrate; Q or Esc to quit.")
    print("=" * 55)

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            both_visible = False

            if (
                results.multi_hand_landmarks
                and results.multi_handedness
            ):
                hand_data = {}

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_style,
                        conn_style
                    )

                    wrist = hand_landmarks.landmark[0]

                    wx = int(wrist.x * w)
                    wy = int(wrist.y * h)

                    opened = is_open_hand(hand_landmarks)

                    hand_data[label] = (
                        wrist.x,
                        wrist.y,
                        wx,
                        wy,
                        opened
                    )

                if "Left" in hand_data and "Right" in hand_data:
                    both_visible = True
                    lost_frames = 0

                    (
                        lx_n,
                        ly_n,
                        lx_px,
                        ly_px,
                        left_open
                    ) = hand_data["Left"]

                    (
                        rx_n,
                        ry_n,
                        rx_px,
                        ry_px,
                        right_open
                    ) = hand_data["Right"]

                    draw_hand_connection(
                        frame,
                        (lx_px, ly_px),
                        (rx_px, ry_px)
                    )

                    angle, direction, strength = (
                        controller.update_steer(
                            (lx_n, ly_n),
                            (rx_n, ry_n)
                        )
                    )

                    if controller.calibrated:
                        throttle_mode = controller.update_throttle(
                            left_open,
                            right_open
                        )

                    else:
                        controller._release("w")
                        controller._release("s")
                        throttle_mode = "NEUTRAL"

                else:
                    lost_frames += 1

                    if lost_frames >= GRACE_FRAMES:
                        controller.release_all()

                        angle = 0.0
                        direction = "STRAIGHT"
                        strength = 0.0
                        throttle_mode = "NEUTRAL"

                        left_open = False
                        right_open = False

            else:
                lost_frames += 1

                if lost_frames >= GRACE_FRAMES:
                    controller.release_all()

                    angle = 0.0
                    direction = "STRAIGHT"
                    strength = 0.0
                    throttle_mode = "NEUTRAL"

                    left_open = False
                    right_open = False

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(
                frame,
                angle,
                direction,
                strength,
                throttle_mode,
                both_visible,
                left_open,
                right_open,
                fps,
                controller.calibrated
            )

            cv2.imshow(
                "Virtual Drive  |  Hand Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            elif key in (ord("c"), ord("C")):
                controller.release_all()

                controller.neutral_angle = None
                controller.calibration_samples.clear()

                angle = 0.0
                direction = "STRAIGHT"
                strength = 0.0
                throttle_mode = "NEUTRAL"

                print(
                    "[INFO] Recalibration started. "
                    "Hold both hands straight."
                )

    finally:
        controller.release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

        print("\n[INFO] Stopped. All keys released.")


if __name__ == "__main__":
    main()