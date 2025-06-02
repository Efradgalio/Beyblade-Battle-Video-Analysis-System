# Beyblade Battle Analyzer 🌀

This project analyzes top-down videos of Beyblade battles using YOLO object detection and BoT-SORT tracking to identify launches, collisions, and winners based on motion analysis.

Example Video Input: [Link Source Video](https://drive.google.com/drive/folders/1EXgGtlmYauc9mDzrxy0B8M3QkyA2JKRZ?usp=sharing)

Example Video Output: [Link Source Video](https://drive.google.com/drive/folders/1hS9j5b7VEfGNih6NgNDgiMAXrnNZeP6u?usp=sharing)

---

## 🚀 Features

- Detects Beyblades using YOLO 11n (You Only Look Once)
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
    Run beyblade_tracker.py in your editor code or notebook

    For Notebook
    ```bash
    analyzer = bt.BeybladeBattleAnalyzer(
    model_path='./runs/detect/train7/weights/best.pt',
    video_path='./source video/beyblade battle 1 clean.mov',
    output_path='./beyblade battle 1 clean tracking.mov'
    )
    analyzer.run_analysis()
    ```

## Output
The output will be a new video analyze by the AI and data summary in CSV file called **battle_summary.csv**.