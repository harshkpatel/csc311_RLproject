import logic2048 as logic

def print_instructions():
    print("Use the following keys to make a move:")
    print("W or w: Move Up")
    print("S or s: Move Down")
    print("A or a: Move Left")
    print("D or d: Move Right")
    print("Q or q: Quit the game")


def main():
    game = logic.Game2048()
    print_instructions()

    while not game.game_end:
        print("\nCurrent Board:")
        print(game)

        move = input("Enter your move (W/S/A/D or Q to quit): ").lower()

        if move == 'q':
            print("Quitting the game. Your final score:", game.get_merge_score())
            break
        elif move in ['w', 's', 'a', 'd']:
            if move == 'w':
                game.make_move(0)  # Move Up
            elif move == 's':
                game.make_move(1)  # Move Down
            elif move == 'a':
                game.make_move(2)  # Move Left
            elif move == 'd':
                game.make_move(3)  # Move Right
        else:
            print("Invalid move! Please use W/S/A/D to make a move or Q to quit.")

    print("Game over! Your final score:", game.get_merge_score())

if __name__ == "__main__":
    main()

