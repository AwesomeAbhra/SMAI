import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(filename="logs/eval_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load datasets
batting_df = pd.read_csv("batting_stints.csv")
bowling_df = pd.read_csv("bowling_overs.csv")

# Handle missing values
batting_df.fillna(0, inplace=True)
bowling_df.fillna(0, inplace=True)

# Load models
with open("models/batting_model.pkl", "rb") as f:
    batting_model = pickle.load(f)
with open("models/wickets_model.pkl", "rb") as f:
    wickets_model = pickle.load(f)
with open("models/runs_model.pkl", "rb") as f:
    runs_model = pickle.load(f)
logging.info("Models loaded from models/")

# Batting Model Evaluation
batting_features = [
    "innings", "over_start", "wickets_fallen", "runs_so_far",
    "batter_avg", "batter_sr", "batter_phase_avg", "batter_phase_sr",
    "bowler_avg", "bowler_er", "bowler_phase_avg", "bowler_phase_er",
    "batter_matchup_avg", "batter_matchup_sr"
]
batting_target = "runs_scored"

batting_df = pd.get_dummies(batting_df, columns=["phase"], prefix="phase")
batting_features += [col for col in batting_df.columns if col.startswith("phase_")]

X_bat = batting_df[batting_features]
y_bat = batting_df[batting_target]
X_bat_train, X_bat_test, y_bat_train, y_bat_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)

batting_rmse = mean_squared_error(y_bat_test, batting_model.predict(X_bat_test), squared=False)
print(f"Batting Model RMSE: {batting_rmse}")
logging.info(f"Batting Model RMSE: {batting_rmse}")

# Bowling Model Evaluation
bowling_features = [
    "innings", "over", "wickets_fallen", "runs_so_far",
    "bowler_avg", "bowler_er", "bowler_phase_avg", "bowler_phase_er",
    "batter1_avg", "batter1_sr", "batter1_phase_avg", "batter1_phase_sr",
    "batter2_avg", "batter2_sr", "batter2_phase_avg", "batter2_phase_sr",
    "bowler_matchup_avg_batter1", "bowler_matchup_sr_batter1",
    "bowler_matchup_avg_batter2", "bowler_matchup_sr_batter2"
]
wickets_target = "wickets_taken"
runs_target = "runs_conceded"

bowling_df = pd.get_dummies(bowling_df, columns=["phase"], prefix="phase")
bowling_features += [col for col in bowling_df.columns if col.startswith("phase_")]

X_bowl = bowling_df[bowling_features]
y_wickets = bowling_df[wickets_target]
y_runs = bowling_df[runs_target]

# Wickets Model
X_bowl_train, X_bowl_test, y_wickets_train, y_wickets_test = train_test_split(X_bowl, y_wickets, test_size=0.2, random_state=42)
wickets_accuracy = accuracy_score(y_wickets_test, wickets_model.predict(X_bowl_test))
print(f"Wickets Model Accuracy: {wickets_accuracy}")
logging.info(f"Wickets Model Accuracy: {wickets_accuracy}")

# Runs Model
X_bowl_train, X_bowl_test, y_runs_train, y_runs_test = train_test_split(X_bowl, y_runs, test_size=0.2, random_state=42)
runs_rmse = mean_squared_error(y_runs_test, runs_model.predict(X_bowl_test), squared=False)
print(f"Runs Model RMSE: {runs_rmse}")
logging.info(f"Runs Model RMSE: {runs_rmse}")