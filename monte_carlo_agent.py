from logic2048 import Game2048
import random, copy
import time
from datetime import timedelta

def random_run(game):
    game_copy = copy.deepcopy(game)
    while not game_copy.game_end: 
        move = random.randint(0, 3)
        game_copy.make_move(move)
    
    return game_copy.get_sum(), game_copy.max_num(), game_copy.get_merge_score()

def monte_carlo_iter(game):
    total_score = [0, 0, 0, 0]

    # For each move (0 - 3)
    for move in range(0,4):
        game_copy = copy.deepcopy(game)
        game_copy.make_move(move)
        if str(game_copy) == str(game):
            continue

        # Try lots of paths with that move using random rollout policy
        for i in range(NUM_ITERS):
            output = random_run(game_copy)   # 0 for largest tile, 1 for sum
            
            # Eval Method 0: Best total sum
            if EVAL_METHOD == 0:
                total_score[move] += output[0]

            # Eval Method 1: Largest tile number
            elif EVAL_METHOD == 1:
                total_score[move] += output[1]

            # Eval Method 2: Largest total merge score
            elif EVAL_METHOD == 2:
                total_score[move] += output[2]

    best_move = total_score.index(max(total_score))
    game.make_move(best_move)

def monte_carlo_run():
    game = Game2048()
    i = 0
    while not game.game_end:
        monte_carlo_iter(game)
        i += 1
    print(game)
    print("Max Square Value: {}".format(game.max_num()))
    print("Total Square Sum: {}".format(game.get_sum()))
    print("Total Merge Score: {}".format(game.get_merge_score()))
    print("--------------------")
    return game.max_num(), game.get_sum(), game.get_merge_score()

def main():
    global NUM_ITERS 
    global NUM_TRIALS 
    global EVAL_METHOD

    NUM_ITERS = 30
    NUM_TRIALS = 10
    EVAL_METHOD = 0
    # Eval method 0: sum of all tiles
    # Eval method 1: largest tiles on board, then sum of all tiles if tied largest tile
    # Eval method 2: merge score
    
    max_val_results = [0] * NUM_TRIALS
    total_sum_results = [0] * NUM_TRIALS
    total_merge_score = [0] * NUM_TRIALS
    
    start_time = time.time()
    for i in range(NUM_TRIALS):
        max_val_results[i], total_sum_results[i], total_merge_score[i] = monte_carlo_run()
    end_time = time.time()
        
    total_sum_avg = sum(total_sum_results) / NUM_TRIALS
    max_val_avg = sum(max_val_results) / NUM_TRIALS
    total_merge_avg = sum(total_merge_score) / NUM_TRIALS

    print("total sum avg: " + str(total_sum_avg))
    print("max val avg: " + str(max_val_avg))
    print("merge score avg: " + str(total_merge_avg))
    print()
    print("time taken: ", str(timedelta(seconds=(end_time - start_time))))

if __name__ == '__main__':
    main()