import pandas as pd
import pickle
import random
from collections import defaultdict
import logging
from utils import compute_player_stats, create_players_df

# Setup logging
logging.basicConfig(filename="logs/infer_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load models
with open("models/batting_model.pkl", "rb") as f:
    batting_model = pickle.load(f)
with open("models/wickets_model.pkl", "rb") as f:
    wickets_model = pickle.load(f)
with open("models/runs_model.pkl", "rb") as f:
    runs_model = pickle.load(f)
logging.info("Models loaded from models/")

# Load batting dataset for phase columns
batting_df = pd.read_csv("batting_stints.csv")
batting_df.fillna(0, inplace=True)
batting_df = pd.get_dummies(batting_df, columns=["phase"], prefix="phase")

# Compute player stats
folder_path = "ipl_json"
batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er, matchup_avg_sr = compute_player_stats(folder_path)

# Get user input for teams
def get_team_inputs():
    print("Enter 11 players for Team A (one per line):")
    team_a = [input().strip() for _ in range(11)]
    print("Enter 11 players for Team B (one per line):")
    team_b = [input().strip() for _ in range(11)]
    batting_team = input("Which team bats first? (A or B): ").strip().upper()
    while batting_team not in ['A', 'B']:
        batting_team = input("Invalid input. Enter A or B: ").strip().upper()
    return team_a, team_b, batting_team

# Get simulation stop point
def get_stop_point():
    innings = int(input("Enter innings (1 or 2): "))
    while innings not in [1, 2]:
        innings = int(input("Invalid input. Enter 1 or 2: "))
    overs = int(input("Enter number of overs (0–20): "))
    while overs < 0 or overs > 20:
        overs = int(input("Invalid input. Enter 0–20: "))
    balls = int(input("Enter number of balls in the over (0–6): "))
    while balls < 0 or balls > 6:
        balls = int(input("Invalid input. Enter 0–6: "))
    return innings, overs, balls

# Probability-based match simulator
def simulate_match(players_df, batting_team_name, innings_num, stop_overs, stop_balls, batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er, matchup_avg_sr):
    team_runs = 0
    wickets = 0
    batting_order = players_df[players_df["team"] == batting_team_name]["name"].tolist()
    current_batsmen = [batting_order[0], batting_order[1]]
    striker = current_batsmen[0]
    dismissed_players = set()
    overs_bowled = defaultdict(int)
    batting_scorecard = defaultdict(lambda: {"runs": 0, "balls": 0, "status": "not out"})
    fielding_team = "Team B" if batting_team_name == "Team A" else "Team A"
    
    # Base probabilities (tuned for ~7.5 runs/over, ~1–2 wickets in 7 overs)
    base_probs = {"0": 0.40, "1": 0.30, "2": 0.10, "3": 0.02, "4": 0.12, "6": 0.04, "out": 0.02}
    
    def get_outcome_probs(batter, bowler, wickets_fallen, phase):
        batter_sr = phase_batting_avg_sr.get(batter, {phase: {"sr": 100}})[phase]["sr"]
        batter_avg = phase_batting_avg_sr.get(batter, {phase: {"avg": 20}})[phase]["avg"]
        bowler_er = phase_bowling_avg_er.get(bowler, {phase: {"er": 8}})[phase]["er"]
        bowler_avg = phase_bowling_avg_er.get(bowler, {phase: {"avg": 30}})[phase]["avg"]
        
        # Adjust based on SR
        sr_factor = batter_sr / 100
        probs = base_probs.copy()
        probs["4"] *= (1 + (sr_factor - 1) * 0.5)
        probs["6"] *= (1 + (sr_factor - 1) * 0.7)
        probs["0"] /= (1 + (sr_factor - 1) * 0.3)
        
        # Adjust based on bowler ER
        er_factor = 8 / bowler_er if bowler_er > 0 else 1
        probs["4"] /= (1 + (er_factor - 1) * 0.4)
        probs["6"] /= (1 + (er_factor - 1) * 0.5)
        probs["0"] *= (1 + (er_factor - 1) * 0.3)
        
        # Adjust out probability
        out_factor = 30 / bowler_avg if bowler_avg > 0 else 1
        probs["out"] *= (1 + (out_factor - 1) * 0.5)
        if batter_avg > 0:
            probs["out"] /= (1 + (batter_avg / 20 - 1) * 0.2)
        probs["out"] *= (1 + wickets_fallen * 0.1)
        
        # Normalize
        total = sum(probs.values())
        for key in probs:
            probs[key] /= total
        return probs

    # Select initial bowler
    primary_bowlers = players_df[(players_df["team"] == fielding_team) & (players_df["is_bowler"])]["name"].tolist()
    current_bowler = random.choice(primary_bowlers) if primary_bowlers else players_df[players_df["team"] == fielding_team]["name"].iloc[-1]
    
    current_over = 0
    balls_in_over = 0
    used_bowlers = [current_bowler]
    while current_over < stop_overs or (current_over == stop_overs and balls_in_over < stop_balls):
        phase = "powerplay" if current_over + 1 <= 6 else "middle" if current_over + 1 <= 16 else "death"
        # Simulate a delivery
        probs = get_outcome_probs(striker, current_bowler, wickets, phase)
        outcomes = list(probs.keys())
        weights = list(probs.values())
        outcome = random.choices(outcomes, weights, k=1)[0]
        
        # Update stats
        if outcome != "out":
            runs = int(outcome)
            team_runs += runs
            batting_scorecard[striker]["runs"] += runs
            batting_scorecard[striker]["balls"] += 1
            if runs % 2 == 1:
                striker = current_batsmen[1] if striker == current_batsmen[0] else current_batsmen[0]
        else:
            wickets += 1
            batting_scorecard[striker]["balls"] += 1
            batting_scorecard[striker]["status"] = "out"
            dismissed_players.add(striker)
            # Get next batsman
            next_batsman_idx = batting_order.index(striker) + 1
            while next_batsman_idx < len(batting_order) and batting_order[next_batsman_idx] in dismissed_players:
                next_batsman_idx += 1
            if next_batsman_idx < len(batting_order) and wickets < 10:
                new_batsman = batting_order[next_batsman_idx]
                current_batsmen[current_batsmen.index(striker)] = new_batsman
                striker = new_batsman
            else:
                break
        
        balls_in_over += 1
        if balls_in_over == 6:
            current_over += 1
            balls_in_over = 0
            overs_bowled[current_bowler] += 1
            striker = current_batsmen[1] if striker == current_batsmen[0] else current_batsmen[0]
            # Select new bowler (not current bowler, max 4 bowlers)
            available_bowlers = players_df[
                (players_df["team"] == fielding_team) & 
                (players_df["is_bowler"]) & 
                (players_df["overs_bowled"] < 4) & 
                (players_df["name"] != current_bowler)
            ]["name"].tolist()
            if available_bowlers and len(used_bowlers) < 4:
                current_bowler = random.choice(available_bowlers)
                if current_bowler not in used_bowlers:
                    used_bowlers.append(current_bowler)
            elif available_bowlers:
                current_bowler = random.choice(available_bowlers)
            else:
                # Fallback to any player with <4 overs (rare)
                available = players_df[
                    (players_df["team"] == fielding_team) & 
                    (players_df["overs_bowled"] < 4) & 
                    (players_df["name"] != current_bowler)
                ]["name"].tolist()
                current_bowler = random.choice(available) if available else current_bowler
            players_df.loc[players_df["name"] == current_bowler, "overs_bowled"] = overs_bowled[current_bowler]
    
    # Update players_df
    players_df["status"] = "not out"
    players_df.loc[players_df["name"].isin(dismissed_players), "status"] = "out"
    
    # Generate scorecard
    phase = "powerplay" if stop_overs + 1 <= 6 else "middle" if stop_overs + 1 <= 16 else "death"
    print("\nScorecard:")
    print(f"{batting_team_name}: {team_runs}/{wickets} ({stop_overs}.{stop_balls} overs)")
    print("\nBatting:")
    for batter, stats in batting_scorecard.items():
        status = f" {stats['status']}" if stats['status'] == "out" else " *"
        print(f"{batter}{status}: {stats['runs']} runs ({stats['balls']} balls)")
    print("\nBowling:")
    for bowler, overs in overs_bowled.items():
        print(f"{bowler}: {overs} overs")
    print(f"\nCurrent Batsmen: {', '.join(current_batsmen)}")
    print(f"Current Bowler: {current_bowler}")
    logging.info(f"Scorecard: {batting_team_name}: {team_runs}/{wickets} ({stop_overs}.{stop_balls} overs)")
    
    match_state = {
        "innings": innings_num,
        "over": stop_overs + stop_balls / 6,
        "phase": phase,
        "runs": team_runs,
        "wickets": wickets,
        "batting_team": batting_team_name,
        "fielding_team": fielding_team,
        "current_batsmen": current_batsmen,
        "current_bowler": current_bowler,
        "bowler_avg": bowling_avg_er.get(current_bowler, {"avg": 30})["avg"],
        "bowler_er": bowling_avg_er.get(current_bowler, {"er": 8})["er"],
        "bowler_phase_avg": phase_bowling_avg_er.get(current_bowler, {phase: {"avg": 30}})[phase]["avg"],
        "bowler_phase_er": phase_bowling_avg_er.get(current_bowler, {phase: {"er": 8}})[phase]["er"],
        "batter1_avg": batting_avg_sr.get(current_batsmen[0], {"avg": 20})["avg"] if current_batsmen else 20,
        "batter1_sr": batting_avg_sr.get(current_batsmen[0], {"sr": 100})["sr"] if current_batsmen else 100,
        "batter1_phase_avg": phase_batting_avg_sr.get(current_batsmen[0], {phase: {"avg": 20}})[phase]["avg"] if current_batsmen else 20,
        "batter1_phase_sr": phase_batting_avg_sr.get(current_batsmen[0], {phase: {"sr": 100}})[phase]["sr"] if current_batsmen else 100,
        "batter2_avg": batting_avg_sr.get(current_batsmen[1], {"avg": 20})["avg"] if len(current_batsmen) > 1 else 20,
        "batter2_sr": batting_avg_sr.get(current_batsmen[1], {"sr": 100})["sr"] if len(current_batsmen) > 1 else 100,
        "batter2_phase_avg": phase_batting_avg_sr.get(current_batsmen[1], {phase: {"avg": 20}})[phase]["avg"] if len(current_batsmen) > 1 else 20,
        "batter2_phase_sr": phase_batting_avg_sr.get(current_batsmen[1], {phase: {"sr": 100}})[phase]["sr"] if len(current_batsmen) > 1 else 100
    }
    
    return match_state, players_df

# Prediction Function
def predict_optimal_orders(match_state, players_df, batting_model, wickets_model, runs_model, batting_avg_sr, phase_batting_avg_sr, matchup_avg_sr):
    phase_columns = [col for col in batting_df.columns if col.startswith("phase_")]
    phase_dict = {col: 0 for col in phase_columns}
    phase_dict[f"phase_{match_state['phase']}"] = 1

    # Batting Prediction
    batting_team_players = players_df[
        (players_df["team"] == match_state["batting_team"]) &
        (players_df["status"] == "not out") &
        (~players_df["name"].isin(match_state["current_batsmen"]))
    ]
    batting_preds = {}
    batting_order = players_df[players_df["team"] == match_state["batting_team"]]["name"].tolist()
    for _, row in batting_team_players.iterrows():
        batter = row["name"]
        order_weight = 1 / (batting_order.index(batter) + 1) if batter in batting_order else 0.1
        stat_weight = row["batter_avg"] / 20 + row["batter_sr"] / 100
        features = {
            "innings": match_state["innings"],
            "over_start": match_state["over"],
            "wickets_fallen": match_state["wickets"],
            "runs_so_far": match_state["runs"],
            "batter_avg": row["batter_avg"],
            "batter_sr": row["batter_sr"],
            "batter_phase_avg": row["batter_phase_avg"][match_state["phase"]]["avg"],
            "batter_phase_sr": row["batter_phase_sr"][match_state["phase"]]["sr"],
            "bowler_avg": match_state["bowler_avg"],
            "bowler_er": match_state["bowler_er"],
            "bowler_phase_avg": match_state["bowler_phase_avg"],
            "bowler_phase_er": match_state["bowler_phase_er"],
            "batter_matchup_avg": matchup_avg_sr.get((batter, match_state["current_bowler"]), {"avg": 20})["avg"],
            "batter_matchup_sr": matchup_avg_sr.get((batter, match_state["current_bowler"]), {"sr": 100})["sr"],
            **phase_dict
        }
        features_df = pd.DataFrame([features])
        pred = batting_model.predict(features_df)[0] * order_weight * stat_weight
        batting_preds[batter] = pred
    next_batsman = max(batting_preds, key=batting_preds.get) if batting_preds else None

    # Bowling Prediction
    bowling_team_players = players_df[
        (players_df["team"] == match_state["fielding_team"]) &
        (players_df["overs_bowled"] < 4) &
        (players_df["is_bowler"]) &
        (players_df["name"] != match_state["current_bowler"])
    ]
    bowling_preds = {}
    for _, row in bowling_team_players.iterrows():
        bowler = row["name"]
        features = {
            "innings": match_state["innings"],
            "over": match_state["over"],
            "wickets_fallen": match_state["wickets"],
            "runs_so_far": match_state["runs"],
            "bowler_avg": row["bowler_avg"],
            "bowler_er": row["bowler_er"],
            "bowler_phase_avg": row["bowler_phase_avg"][match_state["phase"]]["avg"],
            "bowler_phase_er": row["bowler_phase_er"][match_state["phase"]]["er"],
            "batter1_avg": match_state["batter1_avg"],
            "batter1_sr": match_state["batter1_sr"],
            "batter1_phase_avg": match_state["batter1_phase_avg"],
            "batter1_phase_sr": match_state["batter1_phase_sr"],
            "batter2_avg": match_state["batter2_avg"],
            "batter2_sr": match_state["batter2_sr"],
            "batter2_phase_avg": match_state["batter2_phase_avg"],
            "batter2_phase_sr": match_state["batter2_phase_sr"],
            "bowler_matchup_avg_batter1": matchup_avg_sr.get((match_state["current_batsmen"][0], bowler), {"avg": 20})["avg"] if match_state["current_batsmen"] else 20,
            "bowler_matchup_sr_batter1": matchup_avg_sr.get((match_state["current_batsmen"][0], bowler), {"sr": 100})["sr"] if match_state["current_batsmen"] else 100,
            "bowler_matchup_avg_batter2": matchup_avg_sr.get((match_state["current_batsmen"][1], bowler), {"avg": 20})["avg"] if len(match_state["current_batsmen"]) > 1 else 20,
            "bowler_matchup_sr_batter2": matchup_avg_sr.get((match_state["current_batsmen"][1], bowler), {"sr": 100})["sr"] if len(match_state["current_batsmen"]) > 1 else 100,
            **phase_dict
        }
        features_df = pd.DataFrame([features])
        wickets_pred = wickets_model.predict(features_df)[0]
        runs_pred = runs_model.predict(features_df)[0]
        bowling_preds[bowler] = (wickets_pred, runs_pred)
    
    next_bowler = None
    if bowling_preds:
        max_wickets = max(w[0] for w in bowling_preds.values())
        candidates = [(name, runs) for name, (wickets, runs) in bowling_preds.items() if wickets == max_wickets]
        next_bowler = min(candidates, key=lambda x: x[1])[0]

    print(f"\nNext Batsman: {next_batsman}")
    print(f"Next Bowler: {next_bowler}")
    logging.info(f"Next Batsman: {next_batsman}, Next Bowler: {next_bowler}")
    
    return next_batsman, next_bowler

# Main execution
team_a, team_b, batting_team = get_team_inputs()
batting_team_name = "Team A" if batting_team == "A" else "Team B"
players_df = create_players_df(team_a, team_b, batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er)
innings, overs, balls = get_stop_point()

match_state, players_df = simulate_match(players_df, batting_team_name, innings, overs, balls, batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er, matchup_avg_sr)
next_batsman, next_bowler = predict_optimal_orders(match_state, players_df, batting_model, wickets_model, runs_model, batting_avg_sr, phase_batting_avg_sr, matchup_avg_sr)