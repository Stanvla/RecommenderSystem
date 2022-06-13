# %%
import csv
import os
import zipfile
from typing import Tuple, Dict
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from CollaborativeFiltering import CollaborativeFilteringModel
from collections import Counter


def preprocess_ratings(ratings_df: pd.DataFrame, fn: str, csv_fn: str) -> Dict[Tuple[int, int], int]:
    """
    CollaborativeFilteringModel expects `ratings` to be a dict with key (user_id, item_id) and int value.
    Create this dictionary from ratings df.
    :param ratings_df: dataframe with ratings
    :param fn: file name to pickle the result
    :param csv_fn: file where to save ratings_df
    :return: Appropriate dictionary for CollaborativeFilteringModel.
    """
    if not os.path.isfile(fn):
        ratings_dict = dict()
        for idx, row in tqdm(ratings_df.iterrows(), desc='processing ratings', total=ratings_df.shape[0]):
            ratings_dict[(row['user_id'], row['book_id'])] = row['rating']

        with open(fn, 'wb') as f:
            pickle.dump(ratings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        ratings_df.to_csv(csv_fn, index=False)
    else:
        with open(fn, 'rb') as f:
            ratings_dict = pickle.load(f)
    return ratings_dict


def read_data(data_dir: str, csv_zip: str, ratings_pickle: str, csv_fn: str) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int]]:
    """
    Unzip file with data and create dataframes from csvs
    :param data_dir: name of the directory where to extract zip file
    :param csv_zip: path to the zip file
    :param ratings_pickle: path where to store preprocessed ratings, suitable for CollaborativeFilteringModel
    :param csv_fn: where to store processed ratings csv
    :return: Three dataframe `ratings_df` and dictionary `ratings` which is an appropriate dictionary for CollaborativeFilteringModel.
    """
    # read the data if not done yet
    if not os.path.isdir(data_dir):
        with zipfile.ZipFile(csv_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    ratings_df = pd.read_csv(os.path.join(data_dir, 'BX-Book-Ratings.csv'), sep=';', quoting=csv.QUOTE_ALL, encoding='iso-8859-1',)

    # do not user image urls, no need for now
    books_df = pd.read_csv(
        os.path.join(data_dir, 'BX-Books.csv'),
        encoding='iso-8859-1',
        sep=';',
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
    ).drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

    # user contiguous index for isbns
    unique_isbn_rating = ratings_df.ISBN.unique()
    isbn_id_mapping = {isbn: i for i, isbn in enumerate(unique_isbn_rating)}
    ratings_df['book_id'] = ratings_df['ISBN'].map(isbn_id_mapping)

    # add titles
    ratings_df = ratings_df.merge(books_df[['ISBN', 'Book-Title']], on='ISBN', how='left')
    ratings_df['Book-Title'] = ratings_df['Book-Title'].fillna('Unknown Book')

    # use contiguous index for users
    unique_user_ids = ratings_df['User-ID'].unique()
    user_id_mapping = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    ratings_df['User-ID'] = ratings_df['User-ID'].map(user_id_mapping)

    ratings_df = ratings_df.rename(columns={
        'User-ID': 'user_id',
        'Book-Rating': 'rating',
        'Book-Title': 'title',
    })

    ratings = preprocess_ratings(ratings_df, ratings_pickle, csv_fn)
    return ratings_df, ratings


def train_dev_test_split(
        ratings_df: pd.DataFrame,
        test_perc: float,
        seed: int = None
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """
    Splits the data, so that in test and dev there are no unknown users.
    :param ratings_df: dataframe with ratings
    :param test_perc: size of the train and dev set
    :param seed: for reproducibility
    :return: Three dictionaries, train_set, dev_set, test_set
    """
    np.random.seed(seed)
    user_counter = Counter(ratings_df.user_id)

    test_size = math.floor(test_perc * ratings_df.shape[0])
    test_set = {}
    used_indices = []

    for i, row in tqdm(ratings_df.sort_values(by='user_id').iterrows(), total=ratings_df.shape[0], desc='Creating test set'):
        if len(test_set) == test_size:
            break
        # sample only elements which appear in the data multiple times
        if user_counter[row.user_id] > 1 and np.random.rand() > 0.5:
            test_set[(row.user_id, row.book_id)] = row.rating
            # decrease counter
            user_counter[row.user_id] -= 1
            used_indices.append(i)

    # filtered_df = ratings_df[~ratings_df.index.isin(used_indices)]
    dev_size = math.floor(test_perc * (ratings_df.shape[0] - len(used_indices)))
    dev_set = {}
    train_set = {}
    used_indices = set(used_indices)
    for i, row in tqdm(ratings_df.iterrows(), total=ratings_df.shape[0], desc='Creating dev set'):
        if i in used_indices:
            continue

        if user_counter[row.user_id] > 1 and len(dev_set) < dev_size and np.random.rand() > 0.5:
            dev_set[(row.user_id, row.book_id)] = row.rating
            user_counter[row.user_id] -= 1
        else:
            train_set[(row.user_id, row.book_id)] = row.rating

    return train_set, dev_set, test_set


# %%
if __name__ == '__main__':
    # %%
    data_dir = '//app/data'
    csv_zip = '/home/stankvla/Projects/Python/RecomederSystems/app/BX-CSV-Dump.zip'
    ratings_pkl = '/home/stankvla/Projects/Python/RecomederSystems/app/data/ratings.pkl'
    ratings_csv = '/home/stankvla/Projects/Python/RecomederSystems/app/data/new_ratings.csv'
    seed = 0xDEAD

    ratings_df, ratings = read_data(
        data_dir=data_dir,
        csv_zip=csv_zip,
        ratings_pickle=ratings_pkl,
        csv_fn=ratings_csv,
    )

    train, dev, test = train_dev_test_split(
        ratings_df=ratings_df,
        test_perc=0.2,
        seed=seed,
    )


    # %%

    m = CollaborativeFilteringModel(
        ratings=train,
        n_users=ratings_df.user_id.max() + 1,
        n_items=ratings_df.book_id.max() + 1,
        n_features=6,
        mean_norm=True,
    )


    rmse, min_loss, cur_loss = m.train(
        lam=1,
        lr=0.01,
        lr_factor=0.8475,
        n_epochs=8,
        seed=seed,
        log_each=2,
    )
    # %%
    rmse_val = m.evaluate_rmse(dev)
    # %%
    lotr = ratings_df[ratings_df.title == 'Lord of the Rings Trilogy']
    prediction = m.predict_new_user_with_history(
        history={lotr.book_id.unique()[0]: 10},
        round=False,
        lam=1,
        lr=0.0001,
        lr_factor=0.8,
        n_epochs=20,
        seed=seed,
        log_each=2,
    )

    # %%
    prediction_title = []
    for i, val in enumerate(prediction):
        if val < 9:
            continue
        title = ratings_df[ratings_df.book_id == i].title.unique()[0]
        prediction_title.append([title, val])

    prediction_title.sort(key=lambda x: x[1])

    for t, v in prediction_title[:50]:
        if t == 'Unknown Book':
            continue
        print(f'{t:60} :: {v:.3f}')

    # %%
