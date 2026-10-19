# Virtual Racing Wheel 🏎️

**Turn your hands into a racing controller.**

Virtual Racing Wheel is a computer-vision project that transforms a webcam into a hands-free racing input system. Using Python, OpenCV, and MediaPipe, it tracks hand movements and gestures to translate physical motion into keyboard controls for racing games.

The goal is simple: explore how computer vision can bridge the gap between human movement and interactive control systems — without needing a physical steering wheel.

---

## ⚡ The Idea

What if you could steer a racing game using nothing but your hands and a webcam?

Virtual Racing Wheel explores that idea through real-time hand tracking, steering-angle estimation, gesture recognition, and keyboard input simulation.

Instead of relying on a physical controller, the system interprets your movements and maps them to familiar racing controls.

## 🎮 Controls

| Hand movement | Input | Action |
|---|---|---|
| Turn the steering wheel left | `A` | Steer left |
| Turn the steering wheel right | `D` | Steer right |
| Close your right hand into a fist | `W` | Accelerate |
| Open your right hand | `S` | Brake |
| Press `C` | — | Recalibrate neutral steering |
| Press `Q` or `Esc` | — | Exit the application |

*Controls depend on successful hand detection and calibration. Keyboard inputs are simulated by the application.*

---

## 🧠 How It Works

The application connects several components into one interactive pipeline.

1. **Video capture** — OpenCV reads frames from the webcam.
2. **Hand tracking** — MediaPipe detects hand landmarks in the video stream.
3. **Steering estimation** — The program estimates steering direction relative to a calibrated neutral position.
4. **Gesture recognition** — Hand landmarks are used to distinguish acceleration and braking gestures.
5. **Input mapping** — Python keyboard-control functionality translates the detected actions into game inputs.
6. **Visual feedback** — An on-screen dashboard communicates steering state and application status.

The result is a real-time control loop connecting physical movement to digital input.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and control system |
| OpenCV | Webcam access and image processing |
| MediaPipe | Real-time hand landmark detection |
| NumPy | Numerical operations |
| pynput | Keyboard input simulation |

---

## 🚀 Getting Started

### 1. Prerequisites

- Windows PC
- Python 3.11
- Webcam
- Git
- A racing game that accepts keyboard controls

### 2. Clone the repository

```bash
git clone https://github.com/HeamakeshBajina/virtual-racing-wheel.git
cd virtual-racing-wheel
```

### 3. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Launch the controller

```powershell
.\.venv311\Scripts\python.exe .\steering_wheel.py
```

### 6. Calibrate and play

- Position your hands where the webcam can see them clearly.
- Allow the steering calibration to complete.
- Keep your neutral hand position steady during calibration.
- Test the controls before entering a race.
- Use `C` to recalibrate when needed.
- Use `Q` or `Esc` to exit.

**Tip:** Good lighting, a clear camera view, and stable hand visibility can improve tracking reliability.

---

## 🔧 Engineering Challenges

A webcam controller is more than a hand-tracking demo. It must turn imperfect visual measurements into stable, responsive input.

The main engineering challenges include:

- **Steering stability:** Reducing unintended direction changes caused by small hand movements.
- **Responsiveness:** Reacting quickly when the steering direction reverses.
- **Calibration:** Establishing a neutral reference position for steering.
- **Gesture recognition:** Distinguishing acceleration and braking gestures reliably.
- **Input safety:** Releasing simulated keys when tracking is lost or the application exits.
- **Latency:** Minimizing the delay between a physical movement and its corresponding game input.

These challenges guide the ongoing development of the project.

---

## 🗺️ Development Roadmap

This is an actively evolving project. Planned improvements will be added as they are implemented and tested.

- [x] Webcam-based hand tracking
- [x] Neutral steering calibration
- [x] Keyboard-based steering and throttle controls
- [x] On-screen steering interface
- [ ] More robust tracking-loss recovery
- [ ] Adjustable steering sensitivity and dead zones
- [ ] Improved gesture recognition across different hand positions
- [ ] Performance and latency measurements
- [ ] Repeatable testing under different lighting conditions
- [ ] Documented experiments comparing control stability across versions

The roadmap distinguishes current functionality from future work; completed items will be refined as testing continues.

---

## 🔬 What Comes Next?

The next stage is to make the controller more measurable, not just more feature-rich.

Future iterations can evaluate questions such as:

- How quickly does the controller respond to a change in steering direction?
- How much does the steering signal fluctuate when the hands remain still?
- How reliably are controls released when a hand leaves the camera frame?
- Which sensitivity and filtering settings produce the best balance between stability and responsiveness?

The objective is to improve the system through repeatable experiments and documented iterations.

---

## 📁 Project Structure

```text
virtual-racing-wheel/
├── steering_wheel.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🤝 Contributing and Feedback

Ideas, bug reports, and suggestions for improving tracking stability, control responsiveness, and usability are welcome.

If you encounter an issue, include your operating system, Python version, relevant error messages, and the steps needed to reproduce it. Please avoid sharing private information or unnecessary camera footage.

---

## 👨‍💻 About the Project

Virtual Racing Wheel is an independent engineering and software project exploring the intersection of computer vision, human-computer interaction, and real-time control.

It is being developed iteratively, with an emphasis on understanding the underlying system, testing improvements, and documenting the engineering process.

**Built with Python. Driven by vision. Improved through iteration.**