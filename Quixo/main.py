import copy
# Jupyter Notebook for Training Q-learning Player in Quixo Game

import random
import numpy as np
from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self, learning_rate, discount_factor, exploration_prob) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_values = {}
        self.currentState = None
        self.currentAction = None
        self.currentListAction = None
        self.learning = True

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:

        if self.learning:

            self.mon_caca(None, game)

            if np.random.rand() < self.exploration_prob:
                valid_moves = self.get_valid_moves(game)
                action_values = self.q_values.get(self.currentState, {action: 0 for action in valid_moves})
                random_action = random.choice(list(action_values))

                self.currentAction = random_action
                self.currentListAction = action_values

                from_pos, move = self.decode_action(random_action)
                return from_pos, move
            else:
                valid_moves = self.get_valid_moves(game)
                action_values = self.q_values.get(self.currentState, {action: 0 for action in valid_moves})
                best_action = max(action_values, key=action_values.get)
                best_quality = action_values.get(best_action)
                foo = [key for key in action_values if action_values.get(key) == best_quality]
                best_action = random.choice(foo)

                self.currentAction = best_action
                self.currentListAction = action_values

                from_pos, move = self.decode_action(best_action)
                return from_pos, move
        if not self.learning:
            state = self.get_state_representation(game)
            valid_moves = self.get_valid_moves(game)
            action_values = self.q_values.get(state, {action: 0 for action in valid_moves})
            best_action = max(action_values, key=action_values.get)

            best_quality = action_values.get(best_action)
            foo = [key for key in action_values if action_values.get(key) == best_quality]
            best_action = random.choice(foo)

            from_pos, move = self.decode_action(best_action)
            return from_pos, move

    def mon_caca(self, booli=None, game: 'Game' = None) -> None:
        if booli is None:
            if self.currentState is None:
                self.currentState = self.get_state_representation(game)
                return None
            else:
                truc = self.get_state_representation(game)
                self.update_q_values(self.currentState, self.currentAction, self.currentListAction, 0, truc)
                self.currentState = truc
            return None
        if booli:
            self.update_q_values(self.currentState, self.currentAction, self.currentListAction, 1)
            self.currentState = None
            self.currentAction = None
            self.currentListAction = None
            return None
        else:
            self.update_q_values(self.currentState, self.currentAction, self.currentListAction, -1)
            self.currentState = None
            self.currentAction = None
            self.currentListAction = None
            return None

    def get_state_representation(self, game: 'Game') -> str:
        return str(game.get_board().flatten().tolist() + [game.get_current_player()])

    def encode_action(self, from_pos: tuple[int, int], move: Move) -> str:
        return f"{from_pos[0]},{from_pos[1]},{move.value}"

    def decode_action(self, action: str) -> tuple[tuple[int, int], Move]:
        parts = action.split(',')
        return (int(parts[0]), int(parts[1])), Move(int(parts[2]))

    def get_valid_moves(self, game: 'Game') -> list[str]:
        valid_moves = []
        for x in range(5):
            for y in range(5):
                for move in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                    if game._board[x, y] == -1 or (x == 0 and move == Move.TOP) or (x == 4 and move == Move.BOTTOM) or \
                            (y == 0 and move == Move.LEFT) or (y == 4 and move == Move.RIGHT):
                        valid_moves.append(self.encode_action((x, y), move))
        return valid_moves

    def get_valid_moves2(self, game: 'Game') -> list[str]:
        valid_moves = []
        for x in range(5):
            for y in range(5):
                if self.__take((x, y), game.get_current_player(), game.get_board()):
                    for move in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                        if self.__slide((x, y), move, game.get_board()):
                            valid_moves.append(self.encode_action((x, y), move))
        return valid_moves

    def update_q_values(self, state: str, bestaction: str, actionlist, reward: int, next_state: str = None) -> None:
        if state not in self.q_values:
            self.q_values[state] = actionlist

        current_q_value = self.q_values.get(state, {}).get(bestaction, 0)
        if next_state is not None:
            max_next_q_value = max(self.q_values.get(next_state, {}).values(), default=0)
        else:
            max_next_q_value = 0
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                    reward + self.discount_factor * max_next_q_value)

        self.q_values[state][bestaction] = new_q_value

    def __take(self, from_pos: tuple[int, int], player_id: int, board: np.ndarray) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
                               # check if it is in the first row
                                   (from_pos[0] == 0 and from_pos[1] < 5)
                                   # check if it is in the last row
                                   or (from_pos[0] == 4 and from_pos[1] < 5)
                                   # check if it is in the first column
                                   or (from_pos[1] == 0 and from_pos[0] < 5)
                                   # check if it is in the last column
                                   or (from_pos[1] == 4 and from_pos[0] < 5)
                               # and check if the piece can be moved by the current player
                           ) and (board[from_pos] < 0 or board[from_pos] == player_id)
        if acceptable:
            board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move, board: np.ndarray) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                    slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                    slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                    slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                    slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                    slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                    slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    board[(from_pos[0], i)] = board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    board[(from_pos[0], i)] = board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                board[(from_pos[0], board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    board[(i, from_pos[1])] = board[(
                        i - 1, from_pos[1])]
                # move the piece up
                board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    board[(i, from_pos[1])] = board[(
                        i + 1, from_pos[1])]
                # move the piece down
                board[(board.shape[0] - 1, from_pos[1])] = piece
        return acceptable


if __name__ == '__main__':
    # Training the Q-learning player
    def train_q_learning_player(player, num_episodes=1000):
        player.q_values = {}
        for _ in range(num_episodes):
            print(_)
            a = Game()
            winnerre = a.play(player, RandomPlayer())
            if winnerre == 0:
                player.mon_caca(True)
            else:
                player.mon_caca(False)


    # training:
    my_player = MyPlayer(0.1, 0.9, 0.1)
    my_player.learning = True
    train_q_learning_player(my_player, num_episodes=10000)

    # game
    num_games = 100
    wins = 0
    my_player.learning = False

    for _ in range(num_games):

        g = Game()

        winner = g.play(my_player, RandomPlayer())
        if winner == 0:
            wins += 1

    winning_rate = wins / num_games
    print(f"Winning Rate: {winning_rate * 100:.2f}%")
