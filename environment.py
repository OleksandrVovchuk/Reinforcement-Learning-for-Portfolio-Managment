import numpy as np

class Environment:

    def __init__(self, data, capital=1e6):
        self.capital = capital
        self.data = data

    def get_state(self, t, lookback, is_cov_matrix=True, is_raw_time_series=False):

        assert lookback <= t

        decision_making_state = self.data.iloc[t - lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t - lookback:t]
            return decision_making_state

    def get_reward(self, action, action_t, reward_t, alpha=0.01):
        data_period = self.data[action_t:reward_t]
        weights = np.array(action)
        returns = data_period.pct_change().dropna()
        P_ret = np.sum(returns.mean() * weights)
        P_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        sharpe = P_ret / P_vol
        sharpe = np.array([sharpe] * len(self.data.columns))
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]

        return np.dot(returns, weights), rew