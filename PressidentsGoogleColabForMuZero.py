import datetime
import os

import numpy
import torch

from .abstract_game import AbstractGame
import random

actionDict = {0: ['pass'], 1: [2]}
num = 2
for i in range(4, 15):
    for j in range(1, 5):
        for three in range(0, 5):
            action = [i] * j
            for numberThrees in range(three):
                action.append(3)
            actionDict[num] = action
            num += 1
for i in range(1, 5):
    three = []
    for j in range(i):
        three.append(3)
    actionDict[num] = three
    num += 1


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 1,
                                  16)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(226))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 10  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = 'random'  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 2  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 121  # Maximum number of moves if game is not finished before
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = True if torch.cuda.is_available() else False  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 121  # Number of game moves to keep for every batch element
        self.td_steps = 121  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_device = "cpu"  # "cpu" / "cuda"
        self.reanalyse_num_gpus = 0  # Number of GPUs to use for the reanalyse, it can be fractional, don't fortget to take the train worker and the selfplay workers into account

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Presidents(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter an action for player: {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter an action: ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = actionDict
        return f"{action_number}. {actions[action_number]}"

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.random_action()


class Presidents:
    def __init__(self, seed):
        self.random = numpy.random.RandomState(seed)
        self.player1hand = []
        self.player0hand = []
        self.currentPlayer = 1
        self.cardsOnTop = []
        self.prevCardsOnTop = []
        self.dealHands()

    def random_action(self):
        return self.random.shuffle(self.legal_actions())[0]

    def to_play(self):
        return 0 if self.currentPlayer == 1 else 1

    def switchPlayer(self):
        self.currentPlayer *= -1

    def reset(self):
        self.player1hand = []
        self.player0hand = []
        self.currentPlayer = 1
        self.cardsOnTop = []
        self.prevCardsOnTop = []
        self.dealHands()
        return self.get_observation()

    def checkIfPlayerOut(self):
        playerOut = False
        if len(self.currentPlayerHand()) == 0:
            playerOut = True
        return playerOut

    def currentPlayerHand(self):
        if self.to_play() == 0:
            return self.player0hand
        elif self.to_play() == 1:
            return self.player1hand

    def nextPlayerHand(self):
        if self.to_play() == 1:
            return self.player0hand
        elif self.to_play() == 0:
            return self.player1hand

    def convertThreesToNotThrees(self, cards):
        if 3 in cards:
            if cards[0] == 3:
                return [14] * len(cards)
            else:
                return [cards[0]] * len(cards)
        else:
            return cards

    def removeCards(self, cards):
        for card in cards:
            self.currentPlayerHand().remove(card)

    def step(self, action):
        if actionDict[action] == ['pass']:
            toSwitchPlayer = True
            self.cardsOnTop = []
            self.prevCardsOnTop = []
        else:
            self.removeCards(actionDict[action])
        if actionDict[action] == [2]:
            toSwitchPlayer = False
            self.cardsOnTop = []
            self.prevCardsOnTop = [2]
        elif actionDict[action] == self.convertThreesToNotThrees(self.cardsOnTop):
            toSwitchPlayer = False
            self.cardsOnTop = []
            self.prevCardsOnTop = [-2]
        elif not actionDict[action] == ['pass']:
            toSwitchPlayer = True
            self.prevCardsOnTop = self.cardsOnTop
            self.cardsOnTop = actionDict[action]

        out = self.checkIfPlayerOut()
        reward = self.get_reward(out)
        observation = self.get_observation()

        if toSwitchPlayer:
            self.switchPlayer()

        return observation, reward, out

    def get_observation(self):
        # ["previous action",
        #  "action you are playing on",
        #  "other person's hand size",
        #  "your hand (13 ints long)"]
        observation = []
        if self.cardsOnTop == []:
            observation.append(-1)
        else:
            observation.append(self.playToAction(self.cardsOnTop))
        if self.prevCardsOnTop == []:
            observation.append(-1)
        elif self.prevCardsOnTop == [-2]:
            observation.append(-2)
        else:
            observation.append(self.playToAction(self.prevCardsOnTop))
        observation.append(len(self.nextPlayerHand()))
        observation += self.makeHandArray()
        observation = numpy.array([[observation]], dtype="int32")
        return observation

    def get_reward(self, out):
        if self.checkIfPlayerOut():
            reward = 1
        else:
            reward = 0
        return reward

    def startingLegalPlays(self):
        uniqueHand = self.currentPlayerHand().copy()
        possiblePlays = []
        appendTwo = False
        for card in uniqueHand:
            while uniqueHand.count(card) > 1:
                uniqueHand.remove(card)
        while 2 in uniqueHand:
            appendTwo = True
            uniqueHand.remove(2)
        while 3 in uniqueHand:
            uniqueHand.remove(3)
        amountOfThrees = self.currentPlayerHand().count(3)
        for card in uniqueHand:
            count = self.currentPlayerHand().count(card)
            for i in range(1, count + 1):
                possiblePlays.append([card] * i)
        morePossiblePlays = []
        for play in possiblePlays:
            for i in range(1, 1 + amountOfThrees):
                morePossiblePlays.append(play + ([3] * i))
        for i in range(1, 1 + amountOfThrees):
            possiblePlays.append([3] * i)
        if appendTwo:
            possiblePlays.append([2])
        possiblePlays += morePossiblePlays
        return possiblePlays

    def playingOn(self):
        typeOfTrick = len(self.cardsOnTop)
        allPossiblePlays = self.startingLegalPlays()
        possiblePlays = []
        for play in allPossiblePlays:
            # normal playing
            if len(play) == typeOfTrick and self.convertThreesToNotThrees(play)[0] > \
                    self.convertThreesToNotThrees(self.cardsOnTop)[0]:
                possiblePlays.append(play)
            # matching
            elif len(play) == typeOfTrick and play == self.convertThreesToNotThrees(self.cardsOnTop):
                possiblePlays.append(play)
        return possiblePlays

    def possiblePlaysToActions(self, possiblePlays):
        actions = []
        for play in possiblePlays:
            actions.append(list(actionDict.keys())[list(actionDict.values()).index(play)])
        return actions

    def playToAction(self, play):
        return list(actionDict.keys())[list(actionDict.values()).index(play)]

    def legal_actions(self):
        legalActions = []
        typeOfTrick = len(self.cardsOnTop)
        if 2 in self.currentPlayerHand():
            legalActions.append(1)
        if typeOfTrick == 0:
            legalActions += self.possiblePlaysToActions(self.startingLegalPlays())
        else:
            legalActions.append(0)
            legalActions += self.possiblePlaysToActions(self.playingOn())
        return legalActions

    def render(self):
        print("Player 0's Hand: ", self.player0hand)
        print("Player 1's Hand: ", self.player1hand)
        print("Cards on top: ", self.cardsOnTop)
        print("Prev cards on top: ", self.prevCardsOnTop)

    def makeHandArray(self):
        hand = self.currentPlayerHand()
        handArray = []
        for i in range(2, 15):
            howMany = 0
            for card in hand:
                if card == i:
                    howMany += 1
            handArray.append(howMany)
        return handArray

    def dealHands(self):
        deck = []
        for i in range(2, 15):
            for j in range(0, 4):
                deck.append(i)
        self.random.shuffle(deck)
        for i in range(0, 13):
            self.player1hand.append(deck.pop())
            self.player0hand.append(deck.pop())