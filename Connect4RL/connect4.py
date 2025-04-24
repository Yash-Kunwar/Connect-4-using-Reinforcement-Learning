import pygame
import numpy as np
import sys
from pygame.locals import *

# Import your existing AI agent and helper functions
from trainC4 import agent, drop_piece, is_terminal_node, score_move, get_heuristic

# Initialize pygame
pygame.init()

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 216, 230)
GRAY = (200, 200, 200)

# Game parameters
ROWS = 6
COLUMNS = 7
INAROW = 4
SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)
AI_SEARCH_DEPTH = 3  # Depth for minimax search

# Screen dimensions
width = COLUMNS * SQUARESIZE
height = (ROWS + 2) * SQUARESIZE  # Extra rows for piece dropping animation and info panel
size = (width, height)

screen = pygame.display.set_mode(size)
pygame.display.set_caption('Connect Four')
game_font = pygame.font.SysFont("Arial", 32)
info_font = pygame.font.SysFont("Arial", 18)

class Config:
    def __init__(self, rows, columns, inarow):
        self.rows = rows
        self.columns = columns
        self.inarow = inarow

class SimpleObs:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class Button:
    def __init__(self, text, x, y, width, height, color, hover_color, text_color=BLACK):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.current_color = color
        self.rect = pygame.Rect(x, y, width, height)
        
    def draw(self):
        # Draw the button
        pygame.draw.rect(screen, self.current_color, self.rect, 0, 10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, 10)  # Border
        
        # Draw the text
        text_surface = game_font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def is_hover(self, pos):
        if self.rect.collidepoint(pos):
            self.current_color = self.hover_color
            return True
        self.current_color = self.color
        return False
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                return True
        return False

def create_board():
    board = np.zeros((ROWS, COLUMNS))
    return board

def draw_board(board):
    # Clear the board area
    pygame.draw.rect(screen, BLACK, (0, SQUARESIZE, width, ROWS * SQUARESIZE))
    
    # Draw the blue board background
    for c in range(COLUMNS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # Create empty circles for places where pieces can go
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    # Draw the pieces - placing them at the bottom of the board
    for c in range(COLUMNS):
        for r in range(ROWS):
            if board[r][c] == 1:  # Player 1 (RED)
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), 
                                             int((r+1)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == 2:  # Player 2 / AI (YELLOW)
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), 
                                               int((r+1)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

def is_valid_location(board, col):
    # Check if the top row of the column is empty
    return board[0][col] == 0

def get_next_open_row(board, col):
    # Find the lowest empty row in the specified column
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1  # Column is full

def animate_dropping_piece(board, row, col, piece):
    """Animate the piece dropping to its position"""
    color = RED if piece == 1 else YELLOW
    
    # Clear the top area where the moving piece is shown
    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
    
    # Start animation from top of the board
    for r in range(0, row + 1):
        # Clear previous position and redraw the blue board with black circle
        if r > 0:
            # Redraw the blue background at the previous position
            pygame.draw.rect(screen, BLUE, (col*SQUARESIZE, (r-1)*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # Redraw the empty circle
            pygame.draw.circle(screen, BLACK, (int(col*SQUARESIZE+SQUARESIZE/2), 
                                           int((r-1)*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        # Draw piece at current position
        pygame.draw.circle(screen, color, (int(col*SQUARESIZE+SQUARESIZE/2), 
                                       int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()
        pygame.time.wait(50)  # Animation speed
    
    # Finally update the actual board data
    board[row][col] = piece

def check_win(board, piece):
    # Check horizontal
    for r in range(ROWS):
        for c in range(COLUMNS-3):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical
    for r in range(ROWS-3):
        for c in range(COLUMNS):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
                
    return False

def draw_text(text, color, y_position=SQUARESIZE/2):
    # Draw text on top of the screen
    label = game_font.render(text, True, color)
    text_rect = label.get_rect(center=(width/2, y_position))
    screen.blit(label, text_rect)
    pygame.display.update()

def calculate_rewards(board, mark, config):
    """Calculate rewards for each possible move"""
    rewards = {}
    valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
    
    # Convert numpy array to what score_move expects
    grid = board.copy()
    
    total_reward = 0
    for col in valid_moves:
        # Use your existing score_move function to calculate the reward
        reward = score_move(grid, col, mark, config, AI_SEARCH_DEPTH)
        rewards[col] = reward
        total_reward += reward
    
    return rewards, total_reward

def display_rewards(rewards, total_reward, active_col=None):
    """Display the rewards for each column and the total reward"""
    # Clear the info panel area
    pygame.draw.rect(screen, BLACK, (0, (ROWS+1)*SQUARESIZE, width, SQUARESIZE))
    
    # Draw total reward
    total_text = info_font.render(f"Total Reward: {total_reward:.2f}", True, WHITE)
    screen.blit(total_text, (10, (ROWS+1)*SQUARESIZE + 10))
    
    # Draw rewards for each column
    for col, reward in rewards.items():
        # Highlight active column if specified
        color = GREEN if col == active_col else WHITE
        text = info_font.render(f"{reward:.2f}", True, color)
        text_rect = text.get_rect(center=(col*SQUARESIZE + SQUARESIZE/2, (ROWS+1)*SQUARESIZE + 50))
        screen.blit(text, text_rect)
        
        # Draw column indicators
        col_text = info_font.render(f"Col {col}", True, LIGHT_BLUE)
        col_rect = col_text.get_rect(center=(col*SQUARESIZE + SQUARESIZE/2, (ROWS+1)*SQUARESIZE + 30))
        screen.blit(col_text, col_rect)
    
    pygame.display.update()

def play_game():
    board = create_board()
    game_over = False
    turn = 0  # 0 for Player 1, 1 for AI
    
    # Create Play Again button (initially hidden)
    play_again_btn = Button("Play Again", width//2 - 100, SQUARESIZE//2 - 25, 200, 50, GREEN, LIGHT_BLUE)
    show_play_again = False
    
    # Initialize rewards
    config = Config(ROWS, COLUMNS, INAROW)
    player_rewards, player_total = calculate_rewards(board, 1, config)
    ai_rewards, ai_total = calculate_rewards(board, 2, config)
    
    # Clear screen and draw initial board
    screen.fill(BLACK)
    draw_board(board)
    display_rewards(player_rewards, player_total)
    
    while True:  # Main game loop
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            mouse_pos = pygame.mouse.get_pos()
            
            # Handle Play Again button if game is over
            if show_play_again:
                play_again_btn.is_hover(mouse_pos)
                play_again_btn.draw()
                
                if event.type == pygame.MOUSEBUTTONDOWN and play_again_btn.is_clicked(mouse_pos, event):
                    # Reset the game
                    board = create_board()
                    game_over = False
                    turn = 0
                    show_play_again = False
                    player_rewards, player_total = calculate_rewards(board, 1, config)
                    
                    # Clear screen and redraw
                    screen.fill(BLACK)
                    draw_board(board)
                    display_rewards(player_rewards, player_total)
                    pygame.display.update()
                    continue
            
            # Handle game play if game is not over
            if not game_over:
                if event.type == pygame.MOUSEMOTION:
                    # Clear the top row (where we show the moving piece)
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    # Draw the moving piece in the top row
                    posx = event.pos[0]
                    col = int(posx // SQUARESIZE)
                    
                    if turn == 0 and 0 <= col < COLUMNS:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                        
                        # Highlight the active column's reward
                        if col in player_rewards:
                            display_rewards(player_rewards, player_total, col)
                    
                    pygame.display.update()
                    
                if event.type == pygame.MOUSEBUTTONDOWN and turn == 0:
                    # Player 1's turn
                    # Clear the top row
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    
                    # Get column from mouse position
                    posx = event.pos[0]
                    col = int(posx // SQUARESIZE)
                    
                    if 0 <= col < COLUMNS and is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        animate_dropping_piece(board, row, col, 1)
                        
                        # Update and display rewards after player's move
                        ai_rewards, ai_total = calculate_rewards(board, 2, config)
                        display_rewards(ai_rewards, ai_total)
                        
                        if check_win(board, 1):
                            draw_text("Player 1 wins!", RED)
                            game_over = True
                            show_play_again = True
                        elif not any(is_valid_location(board, c) for c in range(COLUMNS)):
                            draw_text("It's a draw!", WHITE)
                            game_over = True
                            show_play_again = True
                        else:
                            turn = 1  # Switch to AI turn
        
        # AI's turn
        if turn == 1 and not game_over:
            # Clear the top row
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            
            # Show "AI is thinking" text
            thinking_text = game_font.render("AI is thinking...", True, YELLOW)
            text_rect = thinking_text.get_rect(center=(width/2, SQUARESIZE/2))
            screen.blit(thinking_text, text_rect)
            pygame.display.update()
            
            # Prepare the observation for the AI
            flat_board = board.flatten().tolist()
            obs = SimpleObs(flat_board, 2)
            config = Config(ROWS, COLUMNS, INAROW)
            
            # Get AI's move
            col = agent(obs, config)
            
            # Highlight AI's chosen column reward
            display_rewards(ai_rewards, ai_total, col)
            pygame.time.wait(500)  # Pause to show the highlighted reward
            
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                animate_dropping_piece(board, row, col, 2)
                
                # Update and display rewards after AI's move
                player_rewards, player_total = calculate_rewards(board, 1, config)
                display_rewards(player_rewards, player_total)
                
                if check_win(board, 2):
                    draw_text("AI wins!", YELLOW)
                    game_over = True
                    show_play_again = True
                elif not any(is_valid_location(board, c) for c in range(COLUMNS)):
                    draw_text("It's a draw!", WHITE)
                    game_over = True
                    show_play_again = True
                else:
                    turn = 0  # Switch to Player 1 turn
        
        # Draw Play Again button if game is over
        if show_play_again:
            play_again_btn.draw()
            
        pygame.display.update()
        pygame.time.wait(50)  # Small delay to prevent CPU hogging
            
if __name__ == "__main__":
    play_game()
