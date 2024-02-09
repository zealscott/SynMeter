import numpy as np
import pandas as pd
import yaml
import json 


class RecordPostprocessor:
    def __init__(self):
        pass

    def post_process(self, data: pd.DataFrame, grouping_mapping: dict):
        assert isinstance(data, pd.DataFrame)
        # add for debug
        # print(data)
        # print(grouping_mapping)
        data = self.unbinning_attributes(data)
        data = self.decode_other_attributes(data, grouping_mapping)
        data = self.ensure_types(data)
        return data

    def unbinning_attributes(self, data: pd.DataFrame):
        """unbin the binned attributes
        
        """
        print("unbinning attributes ------------------------>")
        binning_info = self.config['numerical_binning']
        # print(binning_info)
        for att, spec_list in binning_info.items():
            [s, t, step] = spec_list
            bins = np.r_[-np.inf, np.arange(int(s), int(t), int(step)), np.inf]

            # remove np.inf
            bins[0] = bins[1] - 1
            bins[-1] = bins[-2] + 2

            values_map = {i: int((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)}
            data[att] = data[att].map(values_map)
        return data


    def decode_other_attributes(self, data: pd.DataFrame, decode_mapping: dict):
        """decode the attributes aside from the grouped ones and binned ones
        as for now, it works for decoding the attributes aside from binned ones since we set grouping info NULL

        """
        print("decode other attributes ------------------------>")
        binning_attr = [attr for attr in self.config['numerical_binning'].keys()]
        for attr, mapping in decode_mapping.items():
            if attr in binning_attr:
                continue
            else:
                mapping = pd.Index(mapping)
                data[attr] = mapping[data[attr]]
        return data

    def ensure_types(self, data: pd.DataFrame):
        """refer to the schema file DATA_TYPE,
        to ensure that the generated dataset have valid data types
        
        """
        from experiment import DATA_TYPE
        with open(DATA_TYPE,'r') as f:
            content = json.load(f)
        COLS = content['dtype']
        for col, data_type in COLS.items():
            data[col] = data[col].astype(data_type)
        return data
