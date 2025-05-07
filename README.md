# Cricket Match Simulator

This repository contains scripts for a machine learning-based cricket match simulator that predicts ball-by-ball outcomes and generates scorecards for T20 matches using Random Forest models.

## Repository Structure

- **`train.py`**: Trains Random Forest models (batting, wickets, runs) on IPL datasets and saves model weights to the `models/` directory.
- **`eval.py`**: Loads model weights and evaluates performance on a test set, reporting metrics like RMSE and accuracy.
- **`infer.py`**: Loads model weights, takes user inputs (team lineups, overs, balls), simulates a match, and displays the scorecard and predictions.


**Download the dataset and model weights**:  
   https://drive.google.com/drive/folders/1VQ3x7SO2A07vK985lAZYmXgM6vKEvTJZ?usp=sharing  
   Place the following files and folders in the repository root:
   - `batting_stints.csv`
   - `bowling_overs.csv`
   - `ipl_json/` folder
   - `models/` folder

## Usage

- **Training**: Train models and save weights
  ```bash
  python train.py
  ```

- **Evaluation**: Evaluate models on the test set
  ```bash
  python eval.py
  ```

- **Inference**: Simulate a match with user inputs
  ```bash
  python infer.py
  ```
