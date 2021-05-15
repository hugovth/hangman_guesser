from typing import Dict, List
import logging
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import sys
from tf_agents.environments import py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

REWARD_DEFAULT = {
    "game_success_reward": 1.00,
    "lose_reward": -1.00,
    "guess_success_reward": 0.02,
    "guess_fail_reward": -0.01,
    "guess_repeat_reward": -0.05,
}
DICTIONARY_PATH_DEFAULT = "properties/words_250000_train.txt"


class HangmanEnvironment(py_environment.PyEnvironment):
    """
    Hangman.
    can accept letter or number from 0 to 25 (0->25)
    """

    def __init__(
        self,
        dictionary_path: str = DICTIONARY_PATH_DEFAULT,
        reward_map: Dict[str, float] = REWARD_DEFAULT,
        life_initial: int = 6,
        seed: int = 42,
    ):

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=25, name="letter"
        )
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(30,), dtype=np.int32, minimum=-1, maximum=26, name="game"
        # )
        self._observation_spec = {'observations': array_spec.BoundedArraySpec( shape=(30,), dtype=np.int32, minimum=-1, maximum=26, name="game"),
                                  'legal_moves': array_spec.ArraySpec(shape=(26,), dtype=np.bool_),}   
        self._state = np.empty(30)
        self._state.fill(-1)

        self._episode_ended = False
        self.logger = logging.getLogger(__name__)
        # 26 letters to be proposed
        # self.action_space = spaces.Discrete(26)
        # # 27 letter (26 + '.'+ '_') ex b.nj.ur___________
        # # 30 maximum size of word
        # # 2 state (to be found or not to be found)
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(28),
        #     spaces.Discrete(30),
        # ))
        self.life_initial = life_initial
        self.words_set = list(set(self._build_dictionary(dictionary_path)))
        self.reward_map = reward_map
        self.seed(seed)
        # self._reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def num_moves(self):
        """Returns the total number of moves in this game (legal or not).
        Returns:
            Integer, number of moves.
        """
        return self.guessed_letters

    @staticmethod
    def _build_dictionary(dictionary_path: str) -> List[str]:
        text_file = open(dictionary_path, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def _success(self, action):
        for index, l in enumerate(self.word_to_guess):
            if action == (ord(l) - 97):
                self._state[index] = action
        legal_moves = self._format_legal_moves(self.guessed_letters)
        observations_and_legal_moves = {'observations': np.array([self._state], dtype=np.int32),
                                        'legal_moves': np.logical_not(legal_moves)}

        # 26 correspond to not found
        if 26 not in self._state:
            self._episode_ended = True
            logging.debug(f"You Found {self.word_to_guess}")
            return ts.termination(
                observations_and_legal_moves,
                self.number_of_life * self.reward_map["game_success_reward"],
            )
        else:
            self.render()
            return ts.transition(
                observations_and_legal_moves,
                reward=self.reward_map["guess_success_reward"],
                discount=1.0,
            )

    def _fail(self):
        self.number_of_life -= 1
        legal_moves = self._format_legal_moves(self.guessed_letters)
        observations_and_legal_moves = {'observations': np.array([self._state], dtype=np.int32),
                                        'legal_moves': np.logical_not(legal_moves)}

        if self.number_of_life == 0:
            logging.debug(f"You did not found {self.word_to_guess}")
            self._episode_ended = True

            return ts.termination(
                observations_and_legal_moves, self.reward_map["lose_reward"]
            )

        else:
            self.render()
            logging.debug(self.word_to_guess)
            return ts.transition(
                observations_and_legal_moves,
                reward=self.reward_map["guess_fail_reward"],
                discount=1.0,
            )

    def _repeated_letter(self, action, action_letter):
        self.number_of_life -= 1
        legal_moves = self._format_legal_moves(self.guessed_letters)
        observations_and_legal_moves = {'observations': np.array([self._state], dtype=np.int32),
                                        'legal_moves': np.logical_not(legal_moves)}
        if self.number_of_life == 0:
            logging.debug(f"You did not found {self.word_to_guess}")
            self._episode_ended = True
            return ts.termination(
                observations_and_legal_moves, self.reward_map["lose_reward"]
            )

        else:
            self.render()
            logging.debug("letter repeated")
            return ts.transition(
                observations_and_legal_moves,
                reward=self.reward_map["guess_repeat_reward"],
                discount=0.9,
            )

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        if int(action) in range(0, 26):
            action = int(action)
        else:
            raise TypeError("input should be int in range 0-25")

        action_letter = chr(action + 97)
        if action not in self.guessed_letters:
            self.guessed_letters.append(action)
        else:
            return self._repeated_letter(action, action_letter)
            # logging.debug("letter repeated")
            # self.number_of_life -= 1
            # return ts.transition(
            #     np.array([self._state], dtype=np.int32),
            #     self.reward_map["guess_repeat_reward"],
            # )
            
        
        if action_letter in self.word_to_guess:
            return self._success(action)
        else:
            return self._fail()

    def _reset(self):
        # state at the start of the game
        self.number_of_life = self.life_initial
        self.word_to_guess = np.random.choice(self.words_set)
        self.guessed_letters = []
        self._state.fill(-1)
        self._state[: len(self.word_to_guess)] = 26
        self._episode_ended = False
        self.render()
        legal_moves = self._format_legal_moves(self.guessed_letters)
        observations_and_legal_moves = {'observations': np.array([self._state], dtype=np.int32),
                                        'legal_moves': np.logical_not(legal_moves)}
        return ts.restart(observations_and_legal_moves)
        
    def _format_legal_moves(self, forbidden_moves, action_dim=26):
        """Returns formatted legal moves.
        This function takes a list of actions and converts it into a fixed size vector
        of size action_dim. If an action is legal, its position is set to 0 and -Inf
        otherwise.
        Ex: legal_moves = [0, 1, 3], action_dim = 5
            returns [0, 0, -Inf, 0, -Inf]
        Args:
        legal_moves: list of legal actions.
        action_dim: int, number of actions.
        Returns:
        a vector of size action_dim.
        """
        new_legal_moves = np.full(action_dim, 0)
        if forbidden_moves:
            new_legal_moves[forbidden_moves] = -1
        return new_legal_moves

    def render(self):
        logging.debug(
            "".join([chr(int(x) + 97) for x in self._state])
            .replace("{", ".")
            .replace("`", "")
        )
        logging.debug(f"Remaining lives :{self.number_of_life}")
        logging.debug(f"guessed letter :{[chr(x+97) for x in self.guessed_letters]}")


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    python_environment = HangmanEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(python_environment)
    # print(tf_env.action_space.n)

    # env.reset()
    episode_ended = False
    tf_env.reset()
    reward = 0
    while not episode_ended:
        # ans = 'empty'
        # while len(ans) > 1:
        letter = input("Guessing letter : ")
        print(letter)
        act = ord(letter) - 97
        next_time_step = tf_env.step(act)
        logging.debug(next_time_step)
        reward += next_time_step.reward.numpy()
        episode_ended = tf_env.current_time_step().is_last()
