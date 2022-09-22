import numpy as np
import math

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
            low=-1, high=1, size=(self.user_count, self.features))
        self.item_features = np.random.uniform(
            low=-1, high=1, size=(self.features, self.item_count))

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
                gradient = 2*(ui_rating-prediction)*row_element
            else:
                # applying gradient descent for feature-item matrix
                col_element = float(u_row[wrt_item_idx])
                gradient = 2*(ui_rating-prediction)*col_element
            return gradient

    def user_feature_gradient(self, user_row, wrt_user_idx):
        """
        Average the gradients of a single user-item row with respect to a single user-feature parameter
        """
        summation = 0
        valid_counter = 0
        for col in range(0, self.item_count):
            if self.invalid_val == self.data[user_row, col]:
                continue
            valid_counter += 1
            summation += self.single_gradient(user_row=user_row,
                                              item_col=col, wrt_user_idx=wrt_user_idx)
        if valid_counter == 0:
            return 0
        return summation/valid_counter

    def item_feature_gradient(self, item_col, wrt_item_idx):
        """
        Average the gradients of a single user-item row with respect to a single feature-item parameter
        """
        summation = 0
        valid_counter = 0
        for row in range(0, self.user_count):
            if self.invalid_val == self.data[row, item_col]:
                continue
            valid_counter += 1
            summation += self.single_gradient(user_row=row,
                                              item_col=item_col, wrt_item_idx=wrt_item_idx)
        if valid_counter == 0:
            return 0

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
