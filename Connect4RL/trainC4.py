
import numpy as np
import random


# calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)
    score = get_heuristic(next_grid, mark, config)
    return score



# helper function for score move - gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


# # Helper fnction for score move - gets heuristic score of the grid
# def get_heuristic(grid, mark, config):
#     num_threes = count_windows(grid, mark, 3, config)
#     num_fours = count_windows(grid, 4, mark, config)
#     num_threes_opp = count_windows(grid, 3, mark%2+1, config)
#     score = num_threes - 1e2*num_threes_opp + 1e6*num_fours
#     return score


# Helper function for minimax: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_twos_opp = count_windows(grid, 2, mark%2+1, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    num_fours_opp = count_windows(grid, 4, mark%2+1, config)
    score = num_threes - 1e1*num_twos_opp - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
    return score


# Uses minimax to calculate value of dropping piece in selected column
def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config)
    return score


# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow


# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonala
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False


# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax(child, depth-1, True, mark, config))
        return value


# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)


# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows


N_STEPS = 4   # Number of steps for minimax search

def agent(obs, config):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)


from kaggle_environments import make, evaluate

env = make("connectx", debug=True)

env.run([agent, "random"])

env.render(mode="ipython")



def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))


# get_win_percentages(agent1=agent, agent2="random", n_rounds=10)



def play_connect_four(agent, rows=6, columns=7, inarow=4):
    """
    Plays a Connect Four game against the specified agent.

    Args:
        agent: The agent function (takes obs, config).
        rows: Number of rows in the board.
        columns: Number of columns in the board.
        inarow: Number of pieces in a row to win.
    """

    # Define a simple config object to pass to the agent
    class Config:
        def __init__(self, rows, columns, inarow):
            self.rows = rows
            self.columns = columns
            self.inarow = inarow

    config = Config(rows, columns, inarow)
    board = [0] * (rows * columns)  # Initialize empty board
    game_over = False
    player_turn = 1  # User is player 1, agent is player 2
    mark = 1 # Current player's mark

    def print_board(board):
        """Prints the board to the console."""
        for r in range(rows):
            print(" ".join(str(board[r * columns + c]) for c in range(columns)))

    def get_valid_moves(board):
        """Gets a list of valid columns."""
        return [c for c in range(columns) if board[c] == 0]

    while not game_over:
        print_board(board)

        if player_turn == 1:  # User's turn
            valid_moves = get_valid_moves(board)
            col = -1
            while col not in valid_moves:
                try:
                    col = int(input(f"Your turn (Player 1), choose a column (0-{columns - 1}): "))
                    if col not in range(columns):
                        print("Invalid column.")
                    elif board[col] != 0:
                        print("Column is full.")
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            # Drop the piece
            for row in range(rows - 1, -1, -1):
                if board[row * columns + col] == 0:
                    board[row * columns + col] = 1
                    break


        else:  # Agent's turn
            class SimpleObs:  # Define SimpleObs here, within the agent's turn block
                def __init__(self, board, mark):
                    self.board = board
                    self.mark = mark
            obs = SimpleObs(board, 2)  # Create a simplified obs for the agent
            col = agent(obs, config)

            print(f"Agent chooses column {col}")
            # Drop the piece
            for row in range(rows - 1, -1, -1):
                if board[row * columns + col] == 0:
                    board[row * columns + col] = 2
                    break

        # Check for win (simplified - you might want to use your check_win function)
        # Basic win check (can be expanded for full win conditions)
        for c in range(columns):
          for r in range(rows):
            # Check if current position is filled
            if board[r * columns + c] != 0:
              # Check horizontal
              if c <= columns - inarow:
                if all(board[r * columns + (c + i)] == board[r * columns + c] for i in range(inarow)):
                  game_over = True
              # Check vertical
              if r <= rows - inarow:
                if all(board[(r + i) * columns + c] == board[r * columns + c] for i in range(inarow)):
                  game_over = True
              # Check positive diagonal
              if c <= columns - inarow and r <= rows - inarow:
                if all(board[(r + i) * columns + (c + i)] == board[r * columns + c] for i in range(inarow)):
                  game_over = True
              # Check negative diagonal
              if c <= columns - inarow and r >= inarow - 1:
                if all(board[(r - i) * columns + (c + i)] == board[r * columns + c] for i in range(inarow)):
                  game_over = True

        if game_over:
            print_board(board)
            if player_turn == 1:
              print("You win!")
            else:
              print("Agent wins!")
        elif 0 not in board:
            print_board(board)
            print("It's a draw!")
            game_over = True

        player_turn = 3 - player_turn  # Switch turns (1 -> 2, 2 -> 1)
        mark = 3 - mark



if __name__ == "__main__":
    play_connect_four(agent)


            
