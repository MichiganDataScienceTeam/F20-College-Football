import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn
import json
import ast


print("Reading data")
df = pd.read_csv("../data/play_by_play_fbs_regular_all.csv")
team_elos = json.load(open("../data/elo_features.json"))
games = pd.read_csv("../data/games.csv")
print("Sampling")
sample = df.sample(100_000).copy()

print("Joining")
joined_games = sample.merge(games, left_on="game_id", right_on="id", how="left")
joined_games["is_home"] = sample.offense == sample.home


def outcome(row):
    return (
        row.home_points > row.away_points
        if row.is_home
        else row.away_points > row.home_points
    )


def time_remaining(row):
    period = int(row.period)
    time = ast.literal_eval(row.clock)
    remaining = ((4 - period) * 15 * 60) + (time["minutes"] * 60) + time["seconds"]
    return remaining


def score(row):
    off_score = int(row.offense_score)
    def_score = int(row.defense_score)
    return off_score - def_score


# Adjusted Score = Score / sqrt(Seconds + 1)
def adjusted_score(row):
    time_remaining = row.time_remaining + 1
    score = row.score
    if time_remaining < 0:
        adj_score = None
    else:
        adj_score = score / np.sqrt(time_remaining)
    return adj_score


def yards_from_own_goal_line(row):
    yards = int(row.yards_to_goal)
    yards_from_own_goal = 100 - yards
    return yards_from_own_goal

def yards_to_go_for_first_down(row):
    return row.distance


print("Time remaining")
sample["time_remaining"] = sample[["period", "clock"]].apply(time_remaining, axis=1)
print("Score")
sample["score"] = sample[["offense_score", "defense_score"]].apply(score, axis=1)
print("Adjusted Score")
sample["adj_score"] = sample[["score", "time_remaining"]].apply(adjusted_score, axis=1)
print("Target")
sample["target"] = joined_games[["is_home", "away_points", "home_points"]].apply(
    outcome, axis=1
)
print("Yards from own goal")
sample["yards_from_own_goal_line"] = sample[["yards_to_goal"]].apply(
    yards_from_own_goal_line, axis=1
)
print("Yards to go")
sample["yards_to_go_for_first_down"] = sample["distance"]
print("ELOs")
sample["defense_elo"] = sample["defense"].apply(team_elos.get)
sample["offense_elo"] = sample["offense"].apply(team_elos.get)

print("Writing file")
sample.to_csv("../data/decision_tree_sample.csv", index=False)
