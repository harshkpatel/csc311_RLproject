import logic2048 as logic
import random
import statistics

def run_naive_games():
    scores = []

    for _ in range(1000):
        game = logic.Game2048()

        while not game.game_end:
            # Choose a random move (W/A/S/D)
            move = random.choice(['w', 'a', 's', 'd'])
            game.make_move(['w', 'a', 's', 'd'].index(move))

        scores.append(game.get_merge_score())

    return scores


if __name__ == "__main__":
    random.seed(311)  # Set a seed for reproducibility
    naive_scores = run_naive_games()

    print("Naive Model Results:")
    print("Average Score:", statistics.mean(naive_scores))
    print("Highest Score:", max(naive_scores))
    print("Lowest Score:", min(naive_scores))
