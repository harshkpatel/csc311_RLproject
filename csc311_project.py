import logic2048 as logic

def print_board(mat):
    for row in mat:
        print(" ".join(str(cell) if cell != 0 else '.' for cell in row))
    print()

def main():
    mat = logic.start_game()

    while True:
        print_board(mat)
        move = input("Enter your move (W/A/S/D) or 'Q' to quit: ").upper()

        if move == 'Q':
            print("Quitting the game. Goodbye!")
            break

        if move in ('W', 'A', 'S', 'D'):
            if move == 'W':
                mat, flag = logic.move_up(mat)
            elif move == 'A':
                mat, flag = logic.move_left(mat)
            elif move == 'S':
                mat, flag = logic.move_down(mat)
            elif move == 'D':
                mat, flag = logic.move_right(mat)

            status = logic.get_current_state(mat)
            print(status)

            if status == 'GAME NOT OVER':
                logic.add_new_2(mat)
            else:
                print("Game over! Final state:")
                print_board(mat)
                break

        else:
            print("Invalid move. Please enter W, A, S, D, or Q.")

if __name__ == "__main__":
    main()
