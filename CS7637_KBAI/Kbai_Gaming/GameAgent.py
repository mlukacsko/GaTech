from Game import Game, Type
from Token import Token
import numpy as np
from random import choice
from collections import defaultdict
from functools import lru_cache


class GameAgent:
    def __init__(self, token: Token):
        """
        Initial call to create your agent.
        This only gets called 1 time and then your agent will play multiple games.
        """
        self._token = token
        self.transposition_table = defaultdict(lambda: None)  # Memoization table


    def token(self):
        return self._token

    def make_move(self, game: Game):
        """
        This is the main driver of the agent. The game controller will call this with an updated game object
        every time the agent is expected to make a move.

        The agent will return a Tuple(int, int) for Tic Tac Toe
        or just an int for all Connect Four games.
        """
        board = game.get_board()
        name = game._game_type.name

        match name:
            case 'TIC_TAC_TOE':
                return self.play_tic_tac_toe(board, game)

            case 'CONNECT_4_BASIC':
                return self.play_connect_4(board, game)

            case 'CONNECT_4_EXTENDED':
                return self.play_connect_4_extended(board, game)

            case 'CONNECT_4_MULTIPLAYER':
                return self.play_connect_4_multiplayer(board, game)

            case 'CONNECT_4_HIDDEN_MULTIPLAYER':
                return self.play_connect_4_hidden_multiplayer(board, game)

            case _:
                raise ValueError("Unsupported game type")

    ### Tic Tac Toe Functions ###

    def play_tic_tac_toe(self, board, game):
        """
        Plays the Tic-Tac-Toe game based on current board and token state.
        Returns (row, col) for Tic-Tac-Toe moves.
        """
        player_token = self._token.value()
        opponent_token = game.player2_token().value()

        # Check for winning move
        for row in range(3):
            for col in range(3):
                if self.is_valid_position(board, row, col):
                    # Make a hypothetical move
                    board[row][col] = player_token
                    if self.winning_move(board, self._token):
                        return row, col  # Take the winning move
                    # Undo the move
                    board[row][col] = ''

        # Check for block opponent's winning move
        for row in range(3):
            for col in range(3):
                if self.is_valid_position(board, row, col):
                    # Simulate opponent's move
                    board[row][col] = opponent_token
                    if self.winning_move(board, game.player2_token()):
                        return row, col  # Block the opponent's winning move
                    # Undo the move
                    board[row][col] = ''

        # Take center if available
        if self.is_valid_position(board, 1, 1):
            return 1, 1

        # Take one of the corners if available
        for row, col in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if self.is_valid_position(board, row, col):
                return row, col

        # Take one of the edges if available
        for row, col in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if self.is_valid_position(board, row, col):
                return row, col

        # If no specific move is found, return a random move
        return self.make_random_move(board)

    def is_valid_position(self, board: np.ndarray, row: int, column: int):
        """
        Returns true if there is no player token at the location specified by the row and column positions.
        """
        try:
            return board[row][column] == ''
        except IndexError:
            return False

    def winning_move(self, board: np.ndarray, token: Token) -> bool:
        """
        Checks if the provided token has 3 consecutive positions in Tic-Tac-Toe.
        """
        # Check horizontal
        for r in range(3):
            if board[r][0] == token.value() and board[r][1] == token.value() and board[r][2] == token.value():
                return True

        # Check vertical
        for c in range(3):
            if board[0][c] == token.value() and board[1][c] == token.value() and board[2][c] == token.value():
                return True

        # Check diagonals
        if board[0][0] == token.value() and board[1][1] == token.value() and board[2][2] == token.value():
            return True
        if board[2][0] == token.value() and board[1][1] == token.value() and board[0][2] == token.value():
            return True

        return False

    def make_random_move(self, board: np.ndarray):
        """
        Returns a random move from available positions for Tic-Tac-Toe.
        """
        empty_coord = np.argwhere(board == '')
        return tuple(choice(empty_coord))

    ### Connect Four Functions ###

    def play_connect_4(self, board, game):
        best_score = -float('inf')
        best_move = -1
        player_token = self._token.value()
        opponent_token = game.player2_token().value()

        for col in range(7):
            if self.is_valid_move(board, col):
                row = self.get_next_open_row(board, col)
                board[row][col] = player_token
                score = self.minimax(board, depth=4, alpha=-float('inf'), beta=float('inf'), maximizing=False, player_token=player_token, opponent_token=opponent_token)
                board[row][col] = ''

                if score > best_score:
                    best_score = score
                    best_move = col

        return best_move if best_move != -1 else self.make_random_connect_4_move(board)


    def minimax(self, board, depth, alpha, beta, maximizing, player_token, opponent_token):
        if self.check_terminal_state(board, opponent_token) or depth == 0:
            return self.evaluate_board(board, opponent_token)

        if maximizing:
            max_eval = -float('inf')
            for col in range(7):
                if self.is_valid_move(board, col):
                    row = self.get_next_open_row(board, col)
                    board[row][col] = player_token  # simulate move
                    eval = self.minimax(board, depth - 1, alpha, beta, False, player_token, opponent_token)
                    board[row][col] = ''  # undo move
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for col in range(7):
                if self.is_valid_move(board, col):
                    row = self.get_next_open_row(board, col)
                    board[row][col] = opponent_token  # simulate opponent's move
                    eval = self.minimax(board, depth - 1, alpha, beta, True, player_token, opponent_token)
                    board[row][col] = ''  # undo move
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def check_terminal_state(self, board, opponent_token):
        # Check if either player has won
        if self.check_win(board, self._token.value()) or self.check_win(board, opponent_token):
            return True

        # Check for board is full, this is a tie
        return all(board[0][col] != '' for col in range(7))

    def check_win(self, board, token):
        # Check horizontal
        for row in range(6):
            for col in range(4):  # Only need to check up to column 3 for horizontal
                if all(board[row][col + i] == token for i in range(4)):
                    return True

        # Check vertical
        for row in range(3):  # Check up to row 2 for vertical
            for col in range(7):
                if all(board[row + i][col] == token for i in range(4)):
                    return True

        # Check diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                if all(board[row + i][col + i] == token for i in range(4)):
                    return True

        # Check diagonal (negative slope)
        for row in range(3):
            for col in range(3, 7):
                if all(board[row + i][col - i] == token for i in range(4)):
                    return True

        return False

    def evaluate_board(self, board, opponent_token):
        if self.check_win(board, self._token.value()):
            return 9999  # Player 1 Winning move
        elif self.check_win(board, opponent_token):
            return -9999  # Opponent winning move
        else:
            return 0  # Neutral state

    def get_next_open_row(self, board, col):
        for row in range(5, -1, -1):
            if board[row][col] == '':
                return row
        return None

    def is_valid_move(self, board, col):
        return board[0][col] == ''

    def make_random_connect_4_move(self, board):
        return -1

    ### Connect Four Extended Functions ###

    def play_connect_4_extended(self, board, game):
        best_score = -float('inf')
        best_move = -1
        player_token = self._token.value()
        opponent_token = game.player2_token().value()

        columns = board.shape[1]
        rows = board.shape[0]
        win_length = game._seq_of_tokens

        # Move ordering: prioritize center moves
        move_order = list(range(columns))
        center_col = columns // 2
        move_order.sort(key=lambda x: abs(x - center_col))

        for col in move_order:
            if self.is_valid_move_extended(board, col):
                row = self.get_next_open_row_extended(board, col)
                board[row][col] = player_token  # Simulate the move
                score = self.minimax_extended(board, depth=4, alpha=-float('inf'), beta=float('inf'),
                                              maximizing=False, player_token=player_token,
                                              opponent_token=opponent_token, rows=rows, columns=columns,
                                              win_length=win_length)
                board[row][col] = ''  # Undo the move

                if self.is_opponent_close_to_winning(board, opponent_token, win_length, rows, columns):
                    score -= 1000  # Penalize risky moves

                if score > best_score:
                    best_score = score
                    best_move = col

        return best_move if best_move != -1 else self.make_random_connect_4_move(board, columns)

    def is_opponent_close_to_winning(self, board, opponent_token, win_length, rows, columns):
        for col in range(columns):
            if self.is_valid_move_extended(board, col):
                row = self.get_next_open_row_extended(board, col)
                board[row][col] = opponent_token  # Simulate opponent move
                if self.check_win_extended(board, opponent_token, win_length, rows, columns):
                    board[row][col] = ''  # Undo move
                    return True  # Opponent can win with this move, so block it
                board[row][col] = ''  # Undo move
        return False

    def minimax_extended(self, board, depth, alpha, beta, maximizing, player_token, opponent_token, rows, columns,
                         win_length):
        if self.check_terminal_state_extended(board, player_token, opponent_token, win_length, rows,
                                              columns) or depth == 0:
            return self.evaluate_board_extended(board, opponent_token, win_length, rows, columns)

        if maximizing:
            max_eval = -float('inf')
            for col in range(columns):
                if self.is_valid_move_extended(board, col):
                    row = self.get_next_open_row_extended(board, col)
                    board[row][col] = player_token  # Simulate move
                    eval = self.minimax_extended(board, depth - 1, alpha, beta, False, player_token, opponent_token,
                                                 rows,
                                                 columns, win_length)
                    board[row][col] = ''  # Undo move
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for col in range(columns):
                if self.is_valid_move_extended(board, col):
                    row = self.get_next_open_row_extended(board, col)
                    board[row][col] = opponent_token  # Simulate opponent's move
                    eval = self.minimax_extended(board, depth - 1, alpha, beta, True, player_token, opponent_token,
                                                 rows,
                                                 columns, win_length)
                    board[row][col] = ''  # Undo move
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval
    def check_terminal_state_extended(self, board, player_token, opponent_token, win_length, rows, columns):
        if self.check_win_extended(board, player_token, win_length, rows, columns) or \
                self.check_win_extended(board, opponent_token, win_length, rows, columns):
            return True

        # Check for tie: If all columns are filled
        return all(board[0][col] != '' for col in range(columns))

    def check_win_extended(self, board, token, win_length, rows, columns):
        # Check horizontal win
        for row in range(rows):
            for col in range(columns - win_length + 1):
                if all(board[row][col + i] == token for i in range(win_length)):
                    return True

        # Check vertical win
        for row in range(rows - win_length + 1):
            for col in range(columns):
                if all(board[row + i][col] == token for i in range(win_length)):
                    return True

        # Check diagonal (positive slope)
        for row in range(rows - win_length + 1):
            for col in range(columns - win_length + 1):
                if all(board[row + i][col + i] == token for i in range(win_length)):
                    return True

        # Check diagonal (negative slope)
        for row in range(rows - win_length + 1):
            for col in range(win_length - 1, columns):
                if all(board[row + i][col - i] == token for i in range(win_length)):
                    return True

        return False

    def get_next_open_row_extended(self, board, col):
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row][col] == '':
                return row
        return None

    def is_valid_move_extended(self, board, col):
        return board[0][col] == ''

    def evaluate_board_extended(self, board, opponent_token, win_length, rows, columns):
        player_token = self._token.value()
        score = 0

        # Check for a winning move
        if self.check_win_extended(board, player_token, win_length, rows, columns):
            return 9999  # Winning move for the player
        elif self.check_win_extended(board, opponent_token, win_length, rows, columns):
            return -9999  # Opponent's winning move

        # Heuristic: Control of the center
        center_col = columns // 2
        center_count = sum([1 for row in range(rows) if board[row][center_col] == player_token])
        score += center_count * 3  # Weight center control higher

        # Heuristic: Two-in-a-row, three-in-a-row for both player and opponent
        score += self.count_potential_lines(board, player_token, win_length, rows, columns) * 10
        score -= self.count_potential_lines(board, opponent_token, win_length, rows, columns) * 8

        return score

    def count_potential_lines(self, board, token, win_length, rows, columns):
        potential_lines = 0

        # Check horizontal potential lines
        for row in range(rows):
            for col in range(columns - win_length + 1):
                line = [board[row][col + i] for i in range(win_length)]
                if self.is_potential_line(line, token, win_length):
                    potential_lines += 1

        # Check vertical potential lines
        for row in range(rows - win_length + 1):
            for col in range(columns):
                line = [board[row + i][col] for i in range(win_length)]
                if self.is_potential_line(line, token, win_length):
                    potential_lines += 1

        # Check diagonal (positive slope) potential lines
        for row in range(rows - win_length + 1):
            for col in range(columns - win_length + 1):
                line = [board[row + i][col + i] for i in range(win_length)]
                if self.is_potential_line(line, token, win_length):
                    potential_lines += 1

        # Check diagonal (negative slope) potential lines
        for row in range(rows - win_length + 1):
            for col in range(win_length - 1, columns):
                line = [board[row + i][col - i] for i in range(win_length)]
                if self.is_potential_line(line, token, win_length):
                    potential_lines += 1

        return potential_lines

    def is_potential_line(self, line, token, win_length):
        return line.count(token) > 0 and line.count('') == (win_length - line.count(token))

    # TODO
    # Connect 4 Multiplayer
    def play_connect_4_multiplayer(self, board, game):
        depth = 3 if np.count_nonzero(board == '') > 30 else 4
        move_order = list(range(10))
        move_order.sort(key=lambda x: abs(x - 5))

        best_move = -1
        best_score = -float('inf')
        for col in move_order:
            if self.is_valid_move(board, col):
                row = self.get_next_open_row(board, col)
                board[row][col] = self._token.value()  # Simulate move
                score = self.minimax_multiplayer(board, depth, alpha=-float('inf'), beta=float('inf'),
                                                 player_token=self._token.value(),
                                                 opponent_tokens=[game.player2_token().value(),
                                                                  game.player3_token().value()],
                                                 maximizing=False)
                board[row][col] = ''  # Undo move
                if score > best_score:
                    best_score = score
                    best_move = col

        return best_move if best_move != -1 else self.make_random_connect_4_move(board)

    def minimax_multiplayer(self, board, depth, alpha, beta, player_token, opponent_tokens, maximizing):
        board_key = self.board_hash(board)
        if (result := self.transposition_table.get(board_key)) is not None:
            return result

        if self.check_terminal_state(board, opponent_tokens) or depth == 0:
            evaluation = self.evaluate_board_multiplayer(board, player_token, opponent_tokens)
            self.transposition_table[board_key] = evaluation
            return evaluation

        best_eval = -float('inf') if maximizing else float('inf')
        for col in range(10):
            if self.is_valid_move(board, col):
                row = self.get_next_open_row(board, col)
                board[row][col] = player_token if maximizing else choice(opponent_tokens)  # Alternate moves
                eval = self.minimax_multiplayer(board, depth - 1, alpha, beta, player_token, opponent_tokens,
                                                not maximizing)
                board[row][col] = ''  # Undo move

                # alpha-beta pruning
                if maximizing:
                    best_eval = max(best_eval, eval)
                    alpha = max(alpha, eval)
                else:
                    best_eval = min(best_eval, eval)
                    beta = min(beta, eval)
                if beta <= alpha:
                    break

        self.transposition_table[board_key] = best_eval
        return best_eval

    @staticmethod
    def board_hash(board):
        return tuple(tuple(row) for row in board)

    def evaluate_board_multiplayer(self, board, player_token, opponent_tokens):
        if self.check_win(board, player_token):
            return 9999  # Winning move for the player
        for opp_token in opponent_tokens:
            if self.check_win(board, opp_token):
                return -9999  # Opponent's winning move

        center_col = 10 // 2
        center_count = sum([1 for row in range(9) if board[row][center_col] == player_token])
        score = center_count * 3  # Center control weighted more
        score += self.count_potential_lines(board, player_token, 4, 6, 7) * 10
        for opp_token in opponent_tokens:
            score -= self.count_potential_lines(board, opp_token, 4, 6, 7) * 8

        return score

    def check_terminal_state(self, board, opponent_tokens):
        if self.check_win(board, self._token.value()) or any(
                self.check_win(board, opp_token) for opp_token in opponent_tokens):
            return True

        # Check for full board (tie)
        return all(board[0][col] != '' for col in range(9))

    def play_connect_4_hidden_multiplayer(self, board, game):
        player_token = self._token.value()
        opponent_tokens = [token for token in ['X', 'O', 'R', 'W'] if token != player_token]
        best_score = -float('inf')
        best_move = -1
        columns = board.shape[1]

        # Centralized move order
        move_order = list(range(columns))
        center_col = columns // 2
        move_order.sort(key=lambda x: abs(x - center_col))

        for col in move_order:
            if self.is_valid_move(board, col):
                row = self.get_next_open_row(board, col)
                board[row][col] = player_token
                score = self.evaluate_hidden_multiplayer_board(board, player_token, opponent_tokens)
                board[row][col] = ''

                if score > best_score:
                    best_score = score
                    best_move = col

        return best_move if best_move != -1 else self.make_random_connect_4_move(board)

    def evaluate_hidden_multiplayer_board(self, board, player_token, opponent_tokens):
        score = 0
        rows, cols = board.shape
        center_col = cols // 2

        # Control of the center column
        center_count = sum(1 for row in range(rows) if board[row][center_col] == player_token)
        score += center_count * 3  # Lower weight for faster evaluation

        # Immediate wins or losses
        if self.check_win(board, player_token):
            return 1000
        for opp_token in opponent_tokens:
            if self.check_win(board, opp_token):
                return -1000

        # Avoid deep heuristics for faster runtime
        return score

    def minimax_hidden(self, board, depth, alpha, beta, maximizing, player_token, opponent_tokens):
        if depth == 0 or self.check_terminal_state(board, opponent_tokens):
            return self.evaluate_hidden_multiplayer_board(board, player_token, opponent_tokens)

        if maximizing:
            max_eval = -float('inf')
            for col in range(board.shape[1]):
                if self.is_valid_move(board, col):
                    row = self.get_next_open_row(board, col)
                    board[row][col] = player_token
                    eval = self.minimax_hidden(board, depth - 1, alpha, beta, False, player_token, opponent_tokens)
                    board[row][col] = ''
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for col in range(board.shape[1]):
                if self.is_valid_move(board, col):
                    row = self.get_next_open_row(board, col)
                    board[row][col] = choice(opponent_tokens)
                    eval = self.minimax_hidden(board, depth - 1, alpha, beta, True, player_token, opponent_tokens)
                    board[row][col] = ''
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

