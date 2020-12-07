datasets_dict = {"LCORE":
                     {"path": "https://swift.dkrz.de/v1/dkrz_02942825-0cab-44f3-ad37-80fd5d2e37e3/FESOM2_data/LCORE",
                      "user_path": "https://swiftbrowser.dkrz.de/objects/FESOM2_data/LCORE",
                      "vars": ["temp", "salt", "a_ice", "m_ice", "ssh", "sst"]},
                 "A01":
                     {"path": "https://swift.dkrz.de/v1/dkrz_02942825-0cab-44f3-ad37-80fd5d2e37e3/FESOM2_data/A01",
                      "user_path": "https://swiftbrowser.dkrz.de/objects/FESOM2_data/A01",
                      "vars": [""]},
                 }


class ZarrDataset:
    def __init__(self, path_url, var_list=[], consolidated=True, dset_attrs={}):
        self.path_url = path_url  # can also be local path remove fsspec inthat case
        self.var_list = var_list
        self.is_consolidated = consolidated
        self.dset_attrs = dset_attrs
    @property
    def merged_dataset(self):
        import fsspec  # delay import
        import xarray as xr
        urls = [self.path_url + "/" + var for var in self.var_list]
        dataset_list = [xr.open_zarr(fsspec.get_mapper(url), consolidated=self.is_consolidated) for url in urls]
        da = xr.merge(dataset_list)
        da.attrs.update(self.dset_attrs)
        return da
    @classmethod
    def from_dict(cls, path_var_dict):
        return cls(path_var_dict['path'], path_var_dict['vars'], dset_attrs={"Dataset URL": path_var_dict['user_path']})

    def load(self):
        return self.merged_dataset


LCORE = ZarrDataset.from_dict(datasets_dict['LCORE'])
A01 = ZarrDataset.from_dict(datasets_dict['A01'])
