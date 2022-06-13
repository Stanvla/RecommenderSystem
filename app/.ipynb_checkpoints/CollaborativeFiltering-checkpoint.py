import numpy as np
from icecream import ic
from typing import Dict, Tuple, Optional, List
from timeit import default_timer as timer



class CollaborativeFilteringModel:
    """
    The CollaborativeFilterModel object creates a recommendation model based on Collaborative Filtering approach,
    and will be trained using gradient descent
    :param ratings: dictionary with (user_id, item_id) key and int value, that tells how user_id likes item_id
    :param n_users: number of users
    :param n_items: number of items
    :param n_features: number of features that will be learned for users and items
    :param mean_norm: boolean, if we want to use mean normalization. Useful when users with no ratings are expected
    """

    def __init__(self, ratings: Dict[Tuple[int, int], int], n_users: int, n_items: int, n_features: int, mean_norm: bool):
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.ratings = ratings
        self.M_users = None
        self.M_items = None
        self.trained = False

        self.means = None
        if mean_norm:
            self.means, self.ratings = self._mean_ratings()

        self._user_to_items = {i: [] for i in range(n_users)}
        self._item_to_users = {i: [] for i in range(n_items)}
        for (user_id, item_id), _ in ratings.items():
            self._user_to_items[user_id].append(item_id)
            self._item_to_users[item_id].append(user_id)

    def _mean_ratings(self):
        sums = [0] * self.n_items
        cnts = [0] * self.n_items

        for (user_id, item_id), y in self.ratings.items():
            sums[item_id] += y
            cnts[item_id] += 1

        means = [0] * self.n_items
        for i, (x, y) in enumerate(zip(sums, cnts)):
            means[i] = 0 if y == 0 else x / y

        # now need to update the ratings
        ratings = {(user_id, item_id): y - means[item_id] for (user_id, item_id), y in self.ratings.items()}
        return means, ratings

    @staticmethod
    def _console_log(epoch, min_loss, curr_loss, interval_time, log_interval, total_time):
        loss_info = f'Epoch {epoch:3}, min MSE loss {min_loss:.3f}, curr MSE loss {curr_loss:.3f},'
        time_info = f'last {log_interval} epochs took {interval_time:.3f} secs, elapsed {total_time / 60:.3f} mins'
        print(loss_info, time_info)

    def train(
            self, lam: float,
            lr: float,
            lr_factor: float,
            n_epochs: int,
            seed: Optional[int] = None,
            use_best: bool = False,
            log_each: int = 100,
            init: bool = True,
            user_indices: List[int] = None,
            item_indices: List[int] = None,
    ) -> Tuple[int, int, int]:
        """
        Train the Collaborative Filtering model using gradient descent.
        :param lam: L2 regularization
        :param lr: learning rate for gradient descent
        :param lr_factor: each epoch the learning rate will be multiplied by this factor
        :param n_epochs: number of epochs to train
        :param seed: can fix seed for reproducibility
        :param use_best: after the training can set weights to the best weights found, according to the training loss
        :param log_each: print info every `log_each` epochs
        :return: Tuple with minimum loss and last loss
        """
        if user_indices is None:
            user_indices = list(range(self.n_users))
        if item_indices is None:
            item_indices = list(range(self.n_items))

        if init:
            np.random.seed(seed)
            self.M_items = np.random.uniform(-0.1, 0.1, (self.n_items, self.n_features))
            self.M_users = np.random.uniform(-0.1, 0.1, (self.n_users, self.n_features))
        else:
            self._check_trained()

        self.trained = True
        train_start = timer()
        start = train_start

        min_loss = self._compute_loss(lam)
        cur_loss = min_loss + 1
        opt_M_users = self.M_users
        opt_M_items = self.M_items

        for epoch in range(n_epochs):
            self._update_weights(lam, lr, user_indices, item_indices)
            if epoch % log_each == 0:
                cur_loss = self._compute_loss(lam)
                self._console_log(epoch, min_loss, cur_loss, timer() - start, log_each, timer() - train_start)
                start = timer()

            if min_loss > cur_loss:
                min_loss = cur_loss
                opt_M_items = self.M_items
                opt_M_users = self.M_users

            lr = lr * lr_factor

        if use_best:
            self.M_users = opt_M_users
            self.M_items = opt_M_items

        metric = self.evaluate_rmse()
        print(f'Train RMSE :: {metric}')
        return metric, min_loss, cur_loss

    def _update_weights(self, lam, lr, user_indices, item_indices):
        # regularization is done for every user, not depending on the training data
        new_M_users = self.M_users.copy()
        if user_indices != []:
            new_M_users[user_indices] -= lr * lam * self.M_users[user_indices]
            # update user weights
            for user_id in user_indices:
            # for user_id, item_lst in self._user_to_items.items():
                item_lst = self._user_to_items[user_id]
                ratings = np.array([self.ratings[(user_id, iid)] for iid in item_lst])
                new_M_users[user_id] -= lr * (self.M_items[item_lst] @ self.M_users[user_id] - ratings) @ self.M_items[item_lst]

        # regularization is done for every item, not depending on the training data
        new_M_items = self.M_items.copy()
        if item_indices != []:
            new_M_items[item_indices] -=  lr * lam * self.M_items[item_indices]
            # update item weights
            # for item_id, user_lst in self._item_to_users.items():
            for item_id in item_indices:
                user_lst = self._item_to_users[item_id]
                ratings = np.array([self.ratings[(uid, item_id)] for uid in user_lst])
                new_M_items[item_id] -= lr * (self.M_users[user_lst] @ self.M_items[item_id] - ratings) @ self.M_users[user_lst]

        self.M_users = new_M_users
        self.M_items = new_M_items

    def evaluate_rmse(self, eval_ratings=None):
        if eval_ratings is None:
            eval_ratings = self.ratings
        return self._compute_loss(lam=0, eval_ratings=eval_ratings) ** 1/2

    def _compute_loss(self, lam, eval_ratings=None):
        if eval_ratings is None:
            eval_ratings = self.ratings

        matrix_reg = lambda m: np.sum(np.linalg.norm(m, axis=1) ** 2)
        reg_term = lam * (matrix_reg(self.M_users) + matrix_reg(self.M_items))

        sse, cnt = 0, 0
        mean_rating, _ = self._mean_ratings()
        for (user_id, item_id), y in eval_ratings.items():

            if user_id >= self.M_users.shape[0]:
                prediction = mean_rating[item_id]
                if self.means is None:
                    prediction *= 2
            else:
                prediction = self.predict_user_item(user_id, item_id)

            if self.means is not None:
                sse += (prediction - y - self.means[item_id]) ** 2
            else:
                sse += (prediction - y) ** 2
            cnt += 1

        return 1/cnt * (sse + reg_term)

    @staticmethod
    def _check_index(idx, matrix, name):
        if matrix is None:
            raise RuntimeError(f'{name} matrix is not initialized')

        if idx >= matrix.shape[0]:
            raise RuntimeError(f'Index out of range, max possible {name} index is {matrix.shape[0]}, but {idx} was given.')
        return True

    def _check_trained(self):
        if not self.trained:
            raise RuntimeError('Model is not trained, train the model first.')
        return self.trained

    def predict_user(self, idx: int, round: bool = False) -> np.ndarray:
        """
        Predicts item ratings of the user with id = `idx`
        :param idx: user id, should be smaller than n_users (passed in the initialization)
        :param round: if need to round the results, otherwise float ratings may be returned
        :return: predicted item ratings of the user as a numpy array
        """
        if self._check_trained() and self._check_index(idx, self.M_users, 'user'):

            if self.means is not None:
                result = self.M_items @ self.M_users[idx] + np.array(self.means)
            else:
                result = self.M_items @ self.M_users[idx]

            return result.round() if round else result

    def predict_item(self, idx: int, round: bool = False) -> np.ndarray:
        """
        Predicts user ratings of specific item wigh id = `idx`
        :param idx: item id, should be smaller than n_items
        :param round: whether to round the results, otherwise may return float ratings
        :return: predicted user ratings of the item as a numpy array
        """
        if self._check_trained() and self._check_index(idx, self.M_items, 'item'):

            if self.means is not None:
                result = self.M_users @ self.M_items[idx] + np.array(self.means)
            else:
                result = self.M_users @ self.M_items[idx]

            return result.round() if round else result

    def predict_user_item(self, idx_user: int, idx_item: int, round: bool = False) -> float:
        """
        Predict rating of a specific item for specific user
        :param idx_user: id of the user who rates the item
        :param idx_item: id of the item to be rated
        :param round: whether to round the result, otherwise a float result may be returned
        :return: rating of an item given by the user
        """
        if self._check_index(idx_item, self.M_items, 'item') and self._check_index(idx_user, self.M_users, 'user'):

            if self.means is not None:
                result = self.M_users[idx_user] @ self.M_items[idx_item] + self.means[idx_item]
            else:
                result = self.M_users[idx_user] @ self.M_items[idx_item]

            return result.round() if round else result

    def predict_new_user(self, round: bool = False) -> np.ndarray:
        """
        Estimate ratings for new user with no data
        :param round: whether to round ratings, or leave in floats
        :return: returns numpy array of ratings for the new user
        """
        if self.means is None:
            means, _ = self._mean_ratings()
            means = np.array(means)
        else:
            means = np.array(self.means)
        return means.round() if round else means

    def predict_new_user_with_history(self, history: Dict[int, int], round: bool, **kwargs) -> np.ndarray:
        self._user_to_items[self.n_users] = []
        for item_id, rating in history.items():
            self._user_to_items[self.n_users].append(item_id)
            self._item_to_users[item_id].append(self.n_users)
            self.ratings[(self.n_users, item_id)] = rating
        self.n_users += 1
        np.random.seed(kwargs['seed'])
        self.M_users = np.r_[self.M_users, np.random.uniform(-0.1, 0.1, (1, self.n_features))]
        self.train(**kwargs, init=False, user_indices=[self.n_users - 1], item_indices=[])
        return self.predict_user(self.n_users - 1, round)


if __name__ == '__main__':
    ratings = {
        (0, 0): 5, (0, 1): 5, (0, 2): 4, (0, 3): 0, (0, 4): 0,
        (1, 0): 5, (1, 1): 4, (1, 2): 4, (1, 3): 0, (1, 4): 0,
        (2, 0): 0, (2, 1): 1, (2, 2): 0, (2, 3): 5, (2, 4): 5,
        (3, 0): 0, (3, 1): 0, (3, 2): 1, (3, 3): 4,  # (3, 4): 5,
    }
    n_users = 5
    n_items = 5
    n_features = 2

    m = CollaborativeFilteringModel(
        ratings=ratings,
        n_users=n_users,
        n_items=n_items,
        n_features=n_features,
        mean_norm=True,
    )

    m.train(
        lam=0.1,
        lr=0.2,
        lr_factor=0.98,
        n_epochs=500,
        seed=0xDEAD,
        log_each=50,
    )
    results = []
    for i in range(5):
        results.append(m.predict_user(i, round=False))

    print(np.stack(results).round().T)
    print(np.stack(results).T)
    print(m.predict_new_user())
    print(m.predict_new_user(round=True))
    print(m.M_users)

    result = m.predict_new_user_with_history(
        history={0: 5},
        round=False,
        lam=1,
        lr=0.1,
        lr_factor=0.9,
        n_epochs=200,
        seed=0xDEAD,
        log_each=20,
    )
    print(result)

    # results = []
    # for i in range(5):
    #     results.append(m.predict_user(i, round=True))
    # print(np.stack(results).T)
