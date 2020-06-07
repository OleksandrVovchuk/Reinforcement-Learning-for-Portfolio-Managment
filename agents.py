import random

import numpy as np
import scipy.optimize as sco
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model

from utils import portfolio


class Agent:

    def __init__(self, portfolio_size, allow_short=False, ):
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size,)

    def get_action(self):
        pass

    def get_optimal_weights(self, returns, loss):
        n_assets = len(returns.columns)

        if self.allow_short:
            bounds = tuple((-1.0, 1.0) for x in range(n_assets))
            constraits = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})
        else:
            bounds = tuple((0.0, 1.0) for x in range(n_assets))
            constraits = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

        optimized = sco.minimize(loss, n_assets * [1.0 / n_assets], method='SLSQP', bounds=bounds, constraints=constraits)

        optimal_weights = optimized['x']

        optimal_weights = self.constrains_fix(optimal_weights)

        return optimal_weights

    def constrains_fix(self, optimal_weights):
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)

        return optimal_weights


class MaximizedSharpeAgent(Agent):

    def get_action(self, returns):
        def loss(weights):
            return -portfolio(returns, weights)[2]

        super(Agent, self).get_optimal_weights(returns, loss)


class MaximizedDecorrelationAgent(Agent):

    def get_action(self, returns):
        def loss(weights):
            weights = np.array(weights)
            return np.sqrt(np.dot(weights.T, np.dot(returns.corr(), weights)))

        super(Agent, self).get_optimal_weights(returns, loss)


class MinimalVarianceAgent(Agent):

    def get_action(self, returns):
        def loss(weights):
            return portfolio(returns, weights)[1] ** 2

        super(Agent, self).get_optimal_weights(returns, loss)


class MaximizedReturnsAgent(Agent):

    def get_action(self, returns):
        def loss(weights):
            return -portfolio(returns, weights)[0]

        super(Agent, self).get_optimal_weights(returns, loss)


class RL_Agent(Agent):

    def __init__(self, portfolio_size, e, e_min, e_decay,
                 is_eval=False, allow_short=True):

        super().__init__(portfolio_size, allow_short=False)
        self.action_size = 3

        self.memory4replay = []
        self.is_eval = is_eval

        self.a = 0.5
        self.g = 0.95
        self.e = e
        self.e_min = e_min
        self.e_decay = e_decay

        self.model = self._model()

    def model(self):

        inputs = Input(shape=self.input_shape)
        x = Flatten()(inputs)
        x = Dense(100, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='elu')(x)
        x = Dropout(0.5)(x)

        predicts = []
        for i in range(self.portfolio_size):
            asset_dense = Dense(self.action_size, activation='linear')(x)
            predicts.append(asset_dense)

        model = Model(inputs=inputs, outputs=predicts)
        model.compile(optimizer='adam', loss='mse')
        return model

    def predicts_to_weights(self, pred, allow_short=False):

        weights = np.zeros(len(pred))
        raw_weights = np.argmax(pred, axis=-1)

        saved_min = None

        for e, r in enumerate(raw_weights):
            if r == 0:  # sit
                weights[e] = 0
            elif r == 1:  # buy
                weights[e] = np.abs(pred[e][0][r])
            else:
                weights[e] = -np.abs(pred[e][0][r])

        if not allow_short:
            weights += np.abs(np.min(weights))
            saved_min = np.abs(np.min(weights))
            saved_sum = np.sum(weights)
        else:
            saved_sum = np.sum(np.abs(weights))

        weights /= saved_sum
        return weights, saved_min, saved_sum

    def get_action(self, state):

        if not self.is_eval and random.random() <= self.e:
            w = np.random.normal(0, 1, size=(self.portfolio_size,))

            saved_min = None

            if not self.allow_short:
                w += np.abs(np.min(w))
                saved_min = np.abs(np.min(w))

            saved_sum = np.sum(w)
            w /= saved_sum
            return w, saved_min, saved_sum

        pred = self.model.predict(np.expand_dims(state.values, 0))
        return self.predicts_to_weights(pred, self.allow_short)

    def expReplay(self, batch_size):

        def weights_to_nn_preds_with_reward(action_weights,
                                            reward,
                                            Q_star=np.zeros((self.portfolio_size, self.action_size))):

            Q = np.zeros((self.portfolio_size, self.action_size))
            for i in range(self.portfolio_size):
                if action_weights[i] == 0:
                    Q[i][0] = reward[i] + self.g * np.max(Q_star[i][0])
                elif action_weights[i] > 0:
                    Q[i][1] = reward[i] + self.g * np.max(Q_star[i][1])
                else:
                    Q[i][2] = reward[i] + self.g * np.max(Q_star[i][2])
            return Q

        def restore_Q_from_weights_and_stats(action):
            action_weights, action_min, action_sum = action[0], action[1], action[2]
            action_weights = action_weights * action_sum
            if action_min != None:
                action_weights = action_weights - action_min
            return action_weights

        for (s, s_, action, reward, done) in self.memory4replay:

            action_weights = restore_Q_from_weights_and_stats(action)
            Q_new_value = weights_to_nn_preds_with_reward(action_weights, reward)
            s, s_ = s.values, s_.values

            if not done:
                # reward + g * Q^*(s_, a_)
                Q_star = self.model.predict(np.expand_dims(s_, 0))
                Q_new_value = weights_to_nn_preds_with_reward(action_weights, reward, np.squeeze(Q_star))

            Q_new_value = [xi.reshape(1, -1) for xi in Q_new_value]
            Q_value = self.model.predict(np.expand_dims(s, 0))
            Q = [np.add(a * (1 - self.a), q * self.a) for a, q in zip(Q_value, Q_new_value)]

            # update current Q function with new optimal value
            self.model.fit(np.expand_dims(s, 0), Q, epochs=1, verbose=0)

        if self.e > self.e_min:
            self.e -= self.e_decay
