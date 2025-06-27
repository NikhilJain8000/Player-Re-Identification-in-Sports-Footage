
# Player Re-Identification in Sports Videos

Hi there! 👋  
This project is my submission, where I tackled the challenge of tracking and consistently identifying players in sports footage — even when they vanish behind others or run off-screen.

---

## 🧠 What This Project Does

In fast-paced sports videos, keeping track of who’s who is tricky. My system:
- Detects players, goalkeepers, and referees using **YOLOv11**
- Assigns each person a unique ID
- Keeps those IDs consistent, even after temporary disappearances
- Uses **both motion and appearance features** to keep things smart

---

## 🏗️ Project Structure

Here’s how things are organized:

```
liat_ai_submission/
├── main.py              # Main script that runs the full pipeline
├── tracker.py           # Logic for tracking players using motion + re-identification
├── reid.py              # Extracts deep visual features from player images
├── models/
│   └── player_ball_v11.pt  # YOLOv11 model file (place it here)
├── output/              # Tracked video output goes here
├── requirements.txt     # Python packages needed
├── README.md            # This file
└── report.md            # Full technical write-up
```

---

## ⚙️ Setup Guide

### 1. Clone and Enter the Repo
```bash
git clone [your-repo-link]
cd liat_ai_submission
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install the Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Necessary Files
- Place the `player_ball_v11.pt` model inside the `models/` folder
- Add the input video (e.g. `15sec_input_720p.mp4`) to the root directory

---

## ▶️ How to Run It

Run the tracker on your video with:

```bash
python main.py
```

You can also pass a custom video and output path:

```bash
python main.py --video_path path/to/video.mp4 --output_dir path/to/save/results
```

It will save an annotated video to `output/tracked_output.mp4`.

---

## 🧪 What You’ll See

In the final output:
- Every player gets a **unique ID** that stays with them
- ID is **maintained** even after occlusion or leaving the frame
- **Confidence scores** show how sure the system is
- **Colored boxes** for easy visual separation
- Ball is shown but not tracked like players

---

## 💻 System Requirements

- Python 3.8+
- GPU (recommended for speed, but it’ll work on CPU too)
- 4GB+ RAM
- ~500MB disk space for everything

---

## 🚀 Cool Features

- Smart **2-stage tracking** (motion + appearance)
- **Hungarian algorithm** for best match-making between players and tracks
- Real-time capable
- Clean, modular code that’s easy to extend

---

## 🤔 Problems? Here's Help

If things go wrong:
- Check if `player_ball_v11.pt` is in the right folder
- Make sure your input video is present and path is correct
- If you don’t have a GPU, don’t worry — it will use CPU automatically

---

## 👨‍💻 About Me

This project was developed by me, Nikhil Jain, as part of my AI Internship application to Liat.ai. It’s a fusion of my love for AI and sports, and I had a blast building it.

---

## 📄 License

This is a submission for an internship assignment and intended for evaluation purposes only.
