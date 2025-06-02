# Beyblade Battle Analyzer 🌀

This project analyzes top-down videos of Beyblade battles using YOLO object detection and BoT-SORT tracking to identify launches, collisions, and winners based on motion analysis.

---

## 🚀 Features

- Detects Beyblades using YOLO (You Only Look Once)
- Tracks multiple objects using BoT-SORT
- Detects events:
  - Launch detection
  - Collision detection
  - Battle Duration
  - Broken Beyblade Detection
- Automatically declares the winner
- Saves results to CSV
- Designed for top-down view 1v1 battles

---

## 📂 Folder Structure
project-root/
├── data/ # (ignored) Raw videos, input files
├── outputs/ # (ignored) Detection and tracking results
├── models/ # (ignored) Trained YOLO weights
├── notebooks/ # Jupyter notebooks (can be ignored in Git)
├── src/ # Core source code
│ ├── detector.py
│ ├── tracker.py
│ └── analyzer.py
├── .gitignore
├── README.md
└── requirements.txt

---

## 📦 Setup & Installation

1. **Clone the repository** 

   ```bash
   git clone https://github.com/Efradgalio/Beyblade-Battle-Video-Analysis-System
   cd your-repo-name
   ``` 
2. **Anaconda Setup**
   ```bash
    conda create --name beyblade_analyzer -file requirements.txt
    conda activate beyblade_analyzer
    ```

3. **Run the script** <br>
    Run enrich_game_dataset.py in your editor code
    ```bash
    python enrich_game_dataset.py
    ```

## Output
The output will be a new video analyze by the AI and data summary in CSV file.