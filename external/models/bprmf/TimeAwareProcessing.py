def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import pickle
import os.path
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy import sparse
from elliot.recommender.knn.item_knn.aiolli_ferrari import AiolliSimilarity


class TimeAwareProcessing(object):

    def __init__(self, data):
        """
        This function applies the item-KNN algorithm to initialize the item similarity matrix.
        """

        self._data = data
        item_knn = AiolliSimilarity(data=self._data, maxk=10)
        item_knn.initialize()
        self.i_s = item_knn.w_sparse.toarray()

    def time_mapping(self):
        """
        This function loads the test dataset, then makes a mapping with the public users of KGTERI.
        The working data are stored in a pandas dataframe. A multiprocessing session
        is started to calculate the P matrices which are the time aware interaction matrices of the
        users. Finally, all the R matrices are processed and stored in a pickle file.
        """

        r_values = []
        shape = self.i_s.shape
        matrix_name = 'user_item_matrix' + ".pk"
        matrix_path = os.path.abspath(os.path.join('./data', 'movielens', 'kgteri', matrix_name))

        # Load the TSV file
        public_df = pd.read_csv('data/movielens/train.tsv', delimiter='\t', header=None).drop(columns=2)
        public_df.columns = ['user', 'item', 'time']

        # Here the mapping makes the dataframe private, so ignore the next two public_df variable names
        public_df['user'] = public_df['user'].map(self._data.public_users)
        public_df['item'] = public_df['item'].map(self._data.public_items)
        private_df = public_df.sort_values(['user', 'time'], ascending=[True, True]).reset_index(drop=True)

        # Saving for each user the interaction's time window with items
        u_t_window = (private_df.groupby('user').max('time') - private_df.groupby('user').min('time'))['time']

        # Creating the items time aware interaction matrix

        # Uncomment the next code to recreate the .pk file with the R matrices

        args = [(private_df.groupby('user').get_group(u), u_t_window[u], u, shape) for u in range(u_t_window.size)]
        n_procs = mp.cpu_count() - 2
        print(f'Running multiprocessing with {n_procs} processes')

        with mp.Pool(n_procs) as pool:
            u_list = pool.starmap(items_processing, args)

        if os.path.exists(matrix_path):
            os.remove(matrix_path)
            calculate_save_r(self.i_s, u_list, matrix_path)
        else:
            calculate_save_r(self.i_s, u_list, matrix_path)

        print('Loading R matrices')

        with open(matrix_path, "rb") as file:
            while True:
                try:
                    r_values.append(pickle.load(file))
                except EOFError:
                    break
        print(f'R matrices loaded from \'{matrix_path}\'')
        r_values = r_values


def calculate_save_r(i_sim, u_list, path):
    """
    This function takes the item similarity matrix (acquired by KNN on all the items)
    and its shape, the list of users and the path in which to save the final R matrix.
    Deeply it applies for each user the R = S + (SxP) formula in which S is the item
    similarity matrix and P is the time aware interaction matrix of the specified user.
    Then for each user, the R matrix and the path are passed to the save_matrix() method
    which stores the matrix in a binary pickle file.
    """

    for u in range(u_list.__len__()):
        # Calculating matrix R = S + (SxP) where S = item similarity matrix for all users
        # and P = time aware item matrix for each user and making each matrix sparse to save space
        r_sparse = sparse.csr_matrix(i_sim + (np.multiply(i_sim, u_list[u].toarray())))

        save_matrix(path, r_sparse)


def items_processing(data, time_window, n, shape):
    """
    This function takes the dataframe of the user u which contains the
    ordered by timestamp columns (user, item, timestamp) and performs the
    P = 1 - [(t_j - t_i)/ T_tot] formula in which t_i is the timestamp of the
    i-th item and t_j is the timestamp of the (i+1)-th item (aka the successor item).
    We define P as the time aware interaction matrix.
    The T_tot factor applies a normalization over the entire timestamp interaction
    session of the user u with the items.
    (T_tot = timestamp_max - timestamp_min ----> of user u)
    """

    n_rows = data.reset_index(drop=True).shape[0]
    print(f'running process for user {n}')
    m = np.zeros(shape)

    for i in range(n_rows):

        # iloc's second argument = 1 indicates the item column in the series (user,item,time)
        item_i = data.iloc[i, 1]
        value_i = data.iloc[i, 2]
        for j in range(i + 1, n_rows):

            item_j = data.iloc[j, 1]
            value_j = data.iloc[j, 2]

            # p value contains the item-item time relation
            p = 1 - ((value_j - value_i) / time_window)
            m[item_i][item_j] = p

    sparse_m = sparse.csr_matrix(m)
    print('Processing done for user: ', n)
    return sparse_m


def save_matrix(file_path, m):
    """
    Store matrix in file by append-mode.
    """

    with open(file_path, 'a+b') as file:
        pickle.dump(m, file)
