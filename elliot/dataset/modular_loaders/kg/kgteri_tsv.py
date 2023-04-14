from types import SimpleNamespace
import typing as t
import numpy as np
import pandas as pd
import pickle
import ast

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


# forse vanno rimappati user e item dopo che sono stati mappati da dataset in questa classe

class KGTERITSVLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.kg = getattr(ns, "kg", None)
        self.user_item_matrix_path = getattr(ns, "user_item_matrix", None)
        self.users = users  # questi da dove vengono presi ?
        self.items = items
        if self.kg is not None:
            self.map_ = self.read_triplets(self.kg)  # KG
        self.user_item_matrix = self.load_pkl()

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users  # solo elem comuni
        self.items = self.items & items
        self.map_ = self.map_[self.map_['subject'].isin(self.items)]

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGTERITSVLoader"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        ns.feature_map = self.map_  # è il kg
        return ns

    def read_triplets(self, file_name):
        return pd.read_csv(file_name, sep='\t', header=None, names=['subject', 'predicate', 'object'])
        #return np.array(kg)

    def load_pkl(self):
        r_list = []
        with open(self.user_item_matrix_path, "rb") as file:
            while True:
                try:
                    r_list.append(pickle.load(file))
                except EOFError:
                    break
        print(f'R matrices loaded from \'{self.user_item_matrix_path}\'')
        return r_list
