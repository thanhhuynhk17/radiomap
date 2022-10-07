import numpy as np
import math

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class Matrix_Factorization:
    def __init__(self, data, data_test, features, invalid_val):
        """
        Init |U x k| and |k x I| matrix (k is number of latent features)
        Data size: |U x I|
        """
        self.data = data
        self.data_test = data_test
        self.features = features
        self.invalid_val = invalid_val
        self.user_count = data.shape[0]
        self.item_count = data.shape[1]
        # init U x k and k x I matrix
        self.user_features = np.random.uniform(
            low=0, high=1, size=(self.user_count, self.features))
        self.item_features = np.random.uniform(
            low=0, high=1, size=(self.features, self.item_count))

    def MSE(self):
        """
        Mean square error function comparing dot product of user-feature row and feature-item column to user-item cell
        """
        matrix_product = np.matmul(self.user_features, self.item_features)
        sum = 0
        for i in range(self.user_count):
            for j in range(self.item_count):
                if self.data[i, j] == self.invalid_val:
                    continue
                sum += (self.data[i, j]-matrix_product[i, j])**2
        return sum

    def RMSE(self):
        """
        Root mean square error function comparing dot product of user-feature row and feature-item column to user-item cell
        """
        return math.sqrt(self.MSE())

    def RMSE_test(self):
        """
        Root mean square error function comparing dot product of user-feature row and feature-item column to user-item cell
        """
        matrix_product = np.matmul(self.user_features, self.item_features)
        sum = 0
        for i in range(self.user_count):
            for j in range(self.item_count):
                if self.data_test[i, j] == self.invalid_val:
                    continue
                sum += (self.data_test[i, j]-matrix_product[i, j])**2
        return math.sqrt(sum)

    def single_gradient(self, user_row, item_col, wrt_user_idx=None, wrt_item_idx=None):
        """
        Compute gradient descent of single user-item cell to a single user-feature or feature-item cell
        """

        if wrt_user_idx != None and wrt_item_idx != None:
            return "Too many elements"
        elif wrt_user_idx == None and wrt_item_idx == None:
            return "Insufficient elements"
        else:
            u_row = self.user_features[user_row, :]
            i_col = self.item_features[:, item_col]
            ui_rating = float(self.data[user_row, item_col])
            prediction = float(np.dot(u_row, i_col))

            if wrt_user_idx != None:
                # applying gradient descent for user-feature matrix
                row_element = float(i_col[wrt_user_idx])
                gradient = (ui_rating-prediction)*row_element
            else:
                # applying gradient descent for feature-item matrix
                col_element = float(u_row[wrt_item_idx])
                gradient = (ui_rating-prediction)*col_element
            return gradient

    def user_feature_gradient(self, user_row, wrt_user_idx):
        """
        Average the gradients of a single user-item row with respect to a single user-feature parameter
        """
        summation = 0
        valid_counter = 0
        for col in range(0, self.item_count):
            if self.invalid_val == self.data[user_row, col]:
                if user_row == 0:
                    print(f'[{user_row},{col}]: ignore')
                continue
            valid_counter += 1
            summation += self.single_gradient(user_row=user_row,
                                              item_col=col, wrt_user_idx=wrt_user_idx)
            if user_row == 0:
                print(f'[{user_row},{col}]: {summation}')

        if valid_counter == 0:
            if user_row == 0:
                print(f'user_row={user_row}: valid_counter={valid_counter}')
            return 0
        if user_row == 0:
            print(f'user_row={user_row}: {summation/valid_counter}++++++++++')
        return summation/valid_counter

    def item_feature_gradient(self, item_col, wrt_item_idx):
        """
        Average the gradients of a single user-item row with respect to a single feature-item parameter
        """
        summation = 0
        valid_counter = 0
        for row in range(0, self.user_count):
            if self.invalid_val == self.data[row, item_col]:
                if item_col == 1:
                    print(f'[{row},{item_col}]: ignore')
                continue
            valid_counter += 1
            summation += self.single_gradient(user_row=row,
                                              item_col=item_col, wrt_item_idx=wrt_item_idx)
            if item_col == 1:
                print(f'[{row},{item_col}]: {summation}')
        if valid_counter == 0:
            if item_col == 1:
                print(f'item_col={item_col}: valid_counter={valid_counter}')
            return 0
        if item_col == 1:
            print(f'item_col={item_col}: {summation/valid_counter}=============')
        return summation/valid_counter

    def update_user_features(self, learning_rate):
        """
        Updates every user-feature parameter according to supplied learning rate
        """
        for i in range(0, self.user_count):
            for j in range(0, self.features):
                self.user_features[i, j] += learning_rate * \
                    self.user_feature_gradient(user_row=i, wrt_user_idx=j)

    def update_item_features(self, learning_rate):
        """
        Updates every feature-item parameter according to supplied learning rate
        """
        for i in range(0, self.features):
            for j in range(0, self.item_count):
                self.item_features[i, j] += learning_rate * \
                    self.item_feature_gradient(item_col=j, wrt_item_idx=i)

    def train_model(self, learning_rate=0.1, iterations=1000):
        """
        Trains model, outputting MSE cost/loss every 50 iterations, using supplied learning rate and iterations
        """
        for i in range(iterations):
            self.update_user_features(learning_rate=learning_rate)
            self.update_item_features(learning_rate=learning_rate)
            if (i+1) % 50 == 0:
                print(f"RMSE in iter {i+1}: ", self.RMSE())
                print(f"RMSE test in iter {i+1}: ", self.RMSE_test())


class Matrix_Factorization_V2:
    def __init__(self, Y_data, K, lam=0.1, Xinit=None, Winit=None, learning_rate=0.5, max_iter=1000, print_every=100, user_based=1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization param
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iter
        self.max_iter = max_iter

        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # no. of users, items, and ratings
        self.n_users = int(np.max(Y_data[:, 0]))+1
        self.n_items = int(np.max(Y_data[:, 1]))+1
        self.n_ratings = Y_data.shape[0]
        # get min,max data
        self.min_data = np.min(Y_data[:,2])
        self.max_data = np.max(Y_data[:,2])

        if Xinit is None:  # new
            self.X = np.full((self.n_items, K),1.0)
        else:  # or from saved data
            self.X = Xinit

        if Winit is None:
            self.W = np.full((K, self.n_users),1.0)
        else:  # from saved data
            self.W = Winit

        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()

    def normalize_pred(self, pred):
        return (pred-self.min_data)/(self.max_data-self.min_data)

    def normalize_Y(self):
        # if self.user_based:
        #     user_col = 0
        #     item_col = 1
        #     n_objects = self.n_users
        # # if we want to normalize based on item, just switch first two columns of data
        # else:  # item based
        #     user_col = 1
        #     item_col = 0
        #     n_objects = self.n_items
        # users = self.Y_raw_data[:, user_col]
        # self.mu = np.zeros((n_objects,))
        # for n in range(n_objects):
        #     # row indices of rating done by user n
        #     # since indices need to be integers, we need to convert
        #     ids = np.where(users == n)[0].astype(np.int32)
        #     # indices of all ratings associated with user n
        #     item_ids = self.Y_data_n[ids, item_col]
        #     # and the corresponding ratings
        #     ratings = self.Y_data_n[ids, 2]
        #     # take mean
        #     m = np.mean(ratings)
        #     if np.isnan(m):
        #         m = 0  # to avoid empty array and nan value
        #     self.mu[n] = m
        #     # normalize
        #     self.Y_data_n[ids, 2] = ratings - self.mu[n]

        self.Y_data_n = self.Y_data_n.astype(float)
        # normalized
        self.Y_data_n[:,2] = self.normalize_pred(self.Y_raw_data[:,2])

    def denormalize_pred(self, pred):
        # denormalized
        denormalized = pred * (self.max_data-self.min_data) + self.min_data
        return denormalized
        
        # return pred


    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(
                self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2

        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') +
                           np.linalg.norm(self.W, 'fro'))
        return L

    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        item_ids = self.Y_data_n[ids, 1].astype(
            np.int32)  # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)

    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:, 1] == item_id)[0]
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)

    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))

    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))

    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print('iter =', it + 1, ', loss =',
                      self.loss(), ', RMSE train =', rmse_train)

    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)

        # if self.user_based:
        #     bias = self.mu[u]
        # else:
        #     bias = self.mu[i]
        # pred = self.X[i, :].dot(self.W[:, u]) + bias
        # # truncate if results are out of range [0, 5]
        # if pred < 0:
        #     return 0
        # if pred > 5:
        #     return 5
        # return pred

        pred = self.X[i, :].dot(self.W[:, u])
        return pred


    # def pred_for_user(self, user_id):
    #     """
    #     predict ratings one user give all unrated items
    #     """
    #     ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
    #     items_rated_by_u = self.Y_data_n[ids, 1].tolist()

    #     y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
    #     predicted_ratings = []
    #     for i in range(self.n_items):
    #         if i not in items_rated_by_u:
    #             predicted_ratings.append((i, y_pred[i]))

    #     return predicted_ratings

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0  # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - self.normalize_pred(rate_test[n, 2]))**2

        RMSE = np.sqrt(SE/n_tests)
        return RMSE

    def get_pred_mtx(self):
        pred_mtx = np.dot(self.X, self.W)
        return self.denormalize_pred(pred_mtx)
