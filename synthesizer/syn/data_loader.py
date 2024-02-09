import json
import os
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import yaml


class DataLoader:
    """Load data, bin some attributes, group some attributes,
    during which encode the attributes' values to categorical indexes,
    encode the remained single attributes likewise,
    remove the identifier attribute,
    (if existing) remove those attributes whose values can be determined by others.

    several marginal generation funtions are also included in the class for use.

    """

    def __init__(self):
        self.private_data = None
        self.all_attrs = []

        self.encode_mapping = {}
        self.decode_mapping = {}

        self.encode_schema = {}

    def load_data(self, private_data, encode_mapping):
        self.private_data = private_data
        self.encode_mapping = encode_mapping

        for attr, encode_mapping in self.encode_mapping.items():
            self.decode_mapping[attr] = sorted(encode_mapping, key=encode_mapping.get)

        for attr, encode_mapping in self.encode_mapping.items():
            # note that here schema means all the valid values of encoded ones
            self.encode_schema[attr] = sorted(encode_mapping.values())

    def obtain_attrs(self):
        """return the list of all attributes' name  except the identifier attribute"""
        self.all_attrs = list(self.private_data.columns)
        return self.all_attrs

    def generate_one_way_marginal(self, records: pd.DataFrame, index_attribute: list):
        """generate marginal for one attribute
        (I guess the recommended arg should be in type of str)

        we first assign a new column 'n' and assign them as 1 for each record in orignal DataFrame
        note that aggfunc means aggrigation function
        and we get counts for each candidate value for the specific index_attribute
        we set fill_value=0 for NaN

        """
        # assign a new column 'n' and assign them as 1 for each record in orignal DataFrame
        # pivot_table is used to get counts for each candidate value for the specific index_attribute
        marginal = records.assign(n=1).pivot_table(
            values="n", index=index_attribute, aggfunc=np.sum, fill_value=0
        )
        # we create new indices which is in ascending order to help create a user-friendly pivot table
        indices = sorted([i for i in self.encode_mapping[index_attribute].values()])
        # and we reindex then fillna(0) means we will fill NaN with 0
        marginal = marginal.reindex(index=indices).fillna(0).astype(np.int32)
        return marginal

    def generate_two_way_marginal(
        self, records: pd.DataFrame, index_attribute: list, column_attribute: list
    ):
        """generate marginal for a pair of attributes

        index_attribute corresponds to row index
        column_attribute corresponds to column index

        """
        marginal = records.assign(n=1).pivot_table(
            values="n",
            index=index_attribute,
            columns=column_attribute,
            aggfunc=np.sum,
            fill_value=0,
        )
        # create a new ordered indices for row and column, just serving for a new display order
        indices = sorted([i for i in self.encode_mapping[index_attribute].values()])
        columns = sorted([i for i in self.encode_mapping[column_attribute].values()])
        marginal = (
            marginal.reindex(index=indices, columns=columns).fillna(0).astype(np.int32)
        )

        # print("*********** generating a two-way marginal *********** ")
        # print("*********** i ******* ", indices)
        # print("*********** j ******* ", columns)
        # print(marginal)
        # print("********** tmp count from the two-way marginal ****** ", np.sum(marginal.values))

        return marginal

    def generate_all_one_way_marginals(self, records: pd.DataFrame):
        """generate all the one-way marginals,
        which simply calls generate_one_way_marginal in every cycle round

        """
        all_attrs = self.obtain_attrs()
        marginals = {}
        for attr in all_attrs:
            marginals[frozenset([attr])] = self.generate_one_way_marginal(records, attr)
        print("------------------------> all one way marginals generated")
        return marginals

    def generate_all_two_way_marginals(self, records: pd.DataFrame):
        """generate all the two-way marginals,
        which simply builds a loop and calls generate_two_way_marginal in every cycle round

        """
        all_attrs = self.obtain_attrs()
        marginals = {}
        for i, attr in enumerate(all_attrs):
            for j in range(i + 1, len(all_attrs)):
                marginals[
                    frozenset([attr, all_attrs[j]])
                ] = self.generate_two_way_marginal(records, attr, all_attrs[j])

        print("------------------------> all two way marginals generated")
        # debug
        tmp_num = np.mean(
            [np.sum(marginal.values) for marginal_att, marginal in marginals.items()]
        )
        print(
            "**************** help debug ************** num of records averaged from all two-way marginals:",
            tmp_num,
        )

        return marginals

    def generate_marginal_by_config(
        self, records: pd.DataFrame, config: dict
    ) -> Tuple[Dict, Dict]:
        """config means those marginals_xxxxx.yaml where define generation details
        1. users manually set config about marginal choosing
        2. automatic way of choosing which marginals TODO

        e.g.
        priv_all_two_way:
          total_eps: 990
        e.g.
        priv_all_one_way:
          total_eps: xxxxx

        """
        marginal_sets = {}
        epss = {}
        for marginal_key, marginal_dict in config.items():
            marginals = {}
            if marginal_key == "priv_all_one_way":
                # merge the returned marginal dictionary
                marginals.update(self.generate_all_one_way_marginals(records))
            elif marginal_key == "priv_all_two_way":
                # merge the returned marginal dictionary
                marginals.update(self.generate_all_two_way_marginals(records))
            else:
                raise NotImplementedError
            epss[marginal_key] = marginal_dict["total_eps"]
            marginal_sets[marginal_key] = marginals
        return marginal_sets, epss
