# Project: Sentinel – A Real-time Anomaly Detector for Server Health

🤖 A real-time anomaly detector for server health, using a One-Class SVM to proactively identify system failures by learning 'normal' behavior.

## What It Does

This project simulates a real-time server monitoring system. It uses a **One-Class Support Vector Machine (SVM)** to learn what "normal" server activity (CPU and Memory usage) looks like.

It then watches a stream of data and instantly flags any activity that falls *outside* of that "normal" boundary, catching potential failures like memory leaks or crashes before they happen.

## The "Wow" Concept
The key insight is that we **only train the model on "good" data**. It learns the boundary of normalcy and can spot a failure it has **never seen before**. This is far more advanced than simple rule-based alerts (e.g., "alert if CPU > 90%").

## How to Run

1.  Ensure you have the required libraries:
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
2.  Run the Python script:
    ```bash
    python your_script_name.py
    ```
