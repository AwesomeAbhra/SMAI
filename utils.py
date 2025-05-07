import pandas as pd
import os
import json
from collections import defaultdict

def compute_player_stats(folder_path):
    batting_stats = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})
    bowling_stats = defaultdict(lambda: {"runs": 0, "balls": 0, "wickets": 0})
    phase_batting_stats = defaultdict(lambda: {"powerplay": {"runs": 0, "balls": 0, "dismissals": 0},
                                              "middle": {"runs": 0, "balls": 0, "dismissals": 0},
                                              "death": {"runs": 0, "balls": 0, "dismissals": 0}})
    phase_bowling_stats = defaultdict(lambda: {"powerplay": {"runs": 0, "balls": 0, "wickets": 0},
                                              "middle": {"runs": 0, "balls": 0, "wickets": 0},
                                              "death": {"runs": 0, "balls": 0, "wickets": 0}})
    matchup_stats = defaultdict(lambda: {"runs": 0, "balls": 0, "dismissals": 0})

    def get_phase(over):
        over_num = int(over) + 1
        if 1 <= over_num <= 6:
            return "powerplay"
        elif 7 <= over_num <= 16:
            return "middle"
        elif 17 <= over_num <= 20:
            return "death"
        return "middle"

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(folder_path, filename), "r") as f:
                    match_data = json.load(f)
                    for innings in match_data["innings"]:
                        for over in innings["overs"]:
                            phase = get_phase(over["over"])
                            for delivery in over.get("deliveries", []):
                                batter = delivery["batter"]
                                bowler = delivery["bowler"]
                                runs = delivery["runs"]["batter"]
                                total_runs = delivery["runs"]["total"]
                                extras = delivery["runs"]["extras"]
                                batting_stats[batter]["runs"] += runs
                                batting_stats[batter]["balls"] += 1 if extras == 0 else 0
                                phase_batting_stats[batter][phase]["runs"] += runs
                                phase_batting_stats[batter][phase]["balls"] += 1 if extras == 0 else 0
                                bowling_stats[bowler]["runs"] += total_runs
                                bowling_stats[bowler]["balls"] += 1 if extras == 0 else 0
                                phase_bowling_stats[bowler][phase]["runs"] += total_runs
                                phase_bowling_stats[bowler][phase]["balls"] += 1 if extras == 0 else 0
                                key = (batter, bowler)
                                matchup_stats[key]["runs"] += runs
                                matchup_stats[key]["balls"] += 1 if extras == 0 else 0
                                if "wickets" in delivery:
                                    player_out = delivery["wickets"][0]["player_out"]
                                    batting_stats[player_out]["dismissals"] += 1
                                    bowling_stats[bowler]["wickets"] += 1
                                    phase_batting_stats[player_out][phase]["dismissals"] += 1
                                    phase_bowling_stats[bowler][phase]["wickets"] += 1
                                    matchup_stats[key]["dismissals"] += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    batting_avg_sr = {}
    bowling_avg_er = {}
    phase_batting_avg_sr = {}
    phase_bowling_avg_er = {}
    matchup_avg_sr = {}

    for batter, stats in batting_stats.items():
        avg = stats["runs"] / stats["dismissals"] if stats["dismissals"] > 0 else stats["runs"]
        sr = (stats["runs"] / stats["balls"] * 100) if stats["balls"] > 0 else 0
        batting_avg_sr[batter] = {"avg": avg, "sr": sr}
        phase_batting_avg_sr[batter] = {}
        for phase in phase_batting_stats[batter]:
            p_stats = phase_batting_stats[batter][phase]
            avg = p_stats["runs"] / p_stats["dismissals"] if p_stats["dismissals"] > 0 else p_stats["runs"]
            sr = (p_stats["runs"] / p_stats["balls"] * 100) if p_stats["balls"] > 0 else 0
            phase_batting_avg_sr[batter][phase] = {"avg": avg, "sr": sr}

    for bowler, stats in bowling_stats.items():
        avg = stats["runs"] / stats["wickets"] if stats["wickets"] > 0 else stats["runs"]
        er = (stats["runs"] / stats["balls"] * 6) if stats["balls"] > 0 else 0
        bowling_avg_er[bowler] = {"avg": avg, "er": er, "balls": stats["balls"]}
        phase_bowling_avg_er[bowler] = {}
        for phase in phase_bowling_stats[bowler]:
            p_stats = phase_bowling_stats[bowler][phase]
            avg = p_stats["runs"] / p_stats["wickets"] if p_stats["wickets"] > 0 else p_stats["runs"]
            er = (p_stats["runs"] / p_stats["balls"] * 6) if p_stats["balls"] > 0 else 0
            phase_bowling_avg_er[bowler][phase] = {"avg": avg, "er": er}

    for (batter, bowler), stats in matchup_stats.items():
        avg = stats["runs"] / stats["dismissals"] if stats["dismissals"] > 0 else stats["runs"]
        sr = (stats["runs"] / stats["balls"] * 100) if stats["balls"] > 0 else 0
        matchup_avg_sr[(batter, bowler)] = {"avg": avg, "sr": sr}

    return batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er, matchup_avg_sr

def create_players_df(team_a, team_b, batting_avg_sr, bowling_avg_er, phase_batting_avg_sr, phase_bowling_avg_er):
    player_data = []
    for player in team_a + team_b:
        team = "Team A" if player in team_a else "Team B"
        # Mark as bowler if bowled >100 balls (primary bowlers)
        is_bowler = bowling_avg_er.get(player, {"balls": 0})["balls"] > 100
        player_data.append({
            "name": player,
            "team": team,
            "status": "not out",
            "overs_bowled": 0,
            "is_bowler": is_bowler,
            "batter_avg": batting_avg_sr.get(player, {"avg": 20})["avg"],
            "batter_sr": batting_avg_sr.get(player, {"sr": 100})["sr"],
            "batter_phase_avg": phase_batting_avg_sr.get(player, {"powerplay": {"avg": 20}, "middle": {"avg": 20}, "death": {"avg": 20}}),
            "batter_phase_sr": phase_batting_avg_sr.get(player, {"powerplay": {"sr": 100}, "middle": {"sr": 100}, "death": {"sr": 100}}),
            "bowler_avg": bowling_avg_er.get(player, {"avg": 30})["avg"],
            "bowler_er": bowling_avg_er.get(player, {"er": 8})["er"],
            "bowler_phase_avg": phase_bowling_avg_er.get(player, {"powerplay": {"avg": 30}, "middle": {"avg": 30}, "death": {"avg": 30}}),
            "bowler_phase_er": phase_bowling_avg_er.get(player, {"powerplay": {"er": 8}, "middle": {"er": 8}, "death": {"er": 8}})
        })
    df = pd.DataFrame(player_data)
    # Ensure at least 4 bowlers per team by marking last players as bowlers if needed
    for team in ["Team A", "Team B"]:
        team_bowlers = df[(df["team"] == team) & (df["is_bowler"])]["name"].tolist()
        if len(team_bowlers) < 4:
            additional_bowlers = df[(df["team"] == team) & (~df["is_bowler"])]["name"].tail(4 - len(team_bowlers)).tolist()
            df.loc[df["name"].isin(additional_bowlers), "is_bowler"] = True
    return df