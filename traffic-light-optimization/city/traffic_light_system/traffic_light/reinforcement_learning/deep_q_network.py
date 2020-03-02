from collections import deque
import random
import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class DeepQNetwork:

    def __init__(self, gamma, epsilon, learning_rate, batch_size, use_memory_palace=False,
                 memory_palace_states_total=1, memory_palace_actions_total=1,
                 actions_total=2, use_previous_model=False,
                 model_path='.', model_name_suffix=''):
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.using_memory_palace = use_memory_palace

        self.actions_total = actions_total

        self.model_path = model_path
        self.model_name = 'model_' + model_name_suffix + '.HDF5'

        self.model = self._build_model()
        if use_previous_model:
            self.load()

        self.memory_palace_states_total = memory_palace_states_total
        self.memory_palace_actions_total = memory_palace_actions_total

        if self.using_memory_palace:
            self.memory = [[deque(maxlen=50)]*self.memory_palace_actions_total]*self.memory_palace_states_total
        else:
            self.memory = deque(maxlen=50*self.memory_palace_states_total*self.memory_palace_actions_total)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actions_total)
        action_values = self.model.predict(state)[0]

        return np.argmax(action_values)

    def remember(self, state, action, reward, next_state, done, memory_palace_state=None, memory_palace_action=None):
        if self.using_memory_palace:
            self.memory[memory_palace_state][memory_palace_action].append((state, action, reward, next_state, done))
        else:
            self.memory.append((state, action, reward, next_state, done))


    def replay(self):
        if self.using_memory_palace:
            # Iterates through every slot of the memory palace
            for state_index in range(self.memory_palace_states_total):
                for action_index in range(self.memory_palace_actions_total):

                    mini_batch_size = min(
                        len(self.memory[state_index][action_index]),
                        int(self.batch_size / self.memory_palace_states_total*self.memory_palace_actions_total)
                    )

                    mini_batch = random.sample(self.memory[state_index][action_index], mini_batch_size)
                    self._replay(mini_batch)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            self._replay(mini_batch)

    def get_memory_size(self):

        if self.using_memory_palace:
            size = 0
            for state_index in range(self.memory_palace_states_total):
                for action_index in range(self.memory_palace_actions_total):
                    size += len(self.memory[state_index][action_index])
        else:
            size = len(self.memory)

        return size

    def save(self):
        self.model.save_weights(self.model_path + '/' + self.model_name)

    def load(self):
        self.model.load_weights(self.model_path + '/' + self.model_name)

    def _replay(self, mini_batch):

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                prediction = self.model.predict(next_state)
                target = (reward + self.gamma * np.amax(prediction[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        # Position P
        input_1 = Input(shape=(8, 8, 1))
        # First layer P
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        # Second layer P
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        # Part of the third layer
        x1 = Flatten()(x1)

        # Speed V
        input_2 = Input(shape=(8, 8, 1))
        # First layer V
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        # Second layer V
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        # Part of the third layer
        x2 = Flatten()(x2)

        # Latest traffic signal state L
        input_3 = Input(shape=(1, 1))
        # Part of the third layer
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        # Third layer
        x = Dense(128, activation='relu')(x)
        # Forth layer
        x = Dense(64, activation='relu')(x)
        # Output layer
        x = Dense(self.actions_total, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model