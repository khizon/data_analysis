import pandas as pd
import numpy as np

class Dataloader():
    def __init__(self, train_paths, test_paths, index, cat_cols):
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.cat_cols = cat_cols
        self.index = index

    def encode_categoricals(self, df):
        for col in self.cat_cols:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.astype('int')
        return df

    def reduce_mem_usage(self, df):
            # Reference: https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516103#2899151
            
            print('--- Reducing memory usage')
            initial_mem_usage = df.memory_usage().sum() / 1024**2
            
            for col in df.columns:
                col_type = df[col].dtype

                if col_type.name in ['category', 'object']:
                    raise ValueError(f"Column '{col}' is of type '{col_type.name}'")

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

            final_mem_usage = df.memory_usage().sum() / 1024**2
            print('------ Memory usage before: {:.2f} MB'.format(initial_mem_usage))
            print('------ Memory usage after: {:.2f} MB'.format(final_mem_usage))
            print('------ Decreased memory usage by {:.1f}%'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

            return df
    
    def load_dataframe(self, paths):
        if isinstance(paths, str):
            return pd.read_csv(paths)
        elif isinstance(paths, list):
            df = pd.concat([pd.read_csv(path, index_col=self.index) for path in paths]).reset_index(drop=True)
            df = df.drop_duplicates(keep="last").reset_index(drop=True)

            return df
        else:
            raise ValueError("paths must be a string or a list of strings")

    def load(self):
        print(f'Loading data')
        train = self.load_dataframe(self.train_paths)
        test = self.load_dataframe(self.test_paths)

        train = self.encode_categoricals(train)
        test = self.encode_categoricals(test)

        train = self.reduce_mem_usage(train)
        test = self.reduce_mem_usage(test)

        return train, test