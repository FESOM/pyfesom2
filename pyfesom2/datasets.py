import xarray as xr
import pyfesom2 as pf
from typing import Sequence
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


def fesom_mesh_to_xr(path: str, alpha: int = 50, beta: int = 15, gamma: int = -90) -> xr.Dataset:
    # nod2d = pd.read_csv(path+"/nod2d.out")
    mesh = pf.load_mesh(path, abg=[alpha, beta, gamma])
    # midx = pd.MultiIndex.from_arrays([mesh.x2,mesh.y2], names=['lon','lat'])
    # with open(path+"/aux3d.out", "rt") as f:
    #    num_levels = int(f.readline())
    # levels = pd.read_csv(path+"/aux3d.out", skiprows=1,
    #                     nrows=num_levels, header=None)
    # dtype=np.float32)
    # num_levels = mesh.nlev-1 # using -1 for now
    # levels = mesh.zlev[:-1]
    # nod2d_data = xr.DataArray(nod2d.index.values.astype(np.int32),
    #                          [('nod2',midx)],name='nod2d')
    # lev_data = xr.DataArray(levels,[levels],dims='nz1',name='depth')
    # nod2d_dataset = xr.merge([nod2d_data,lev_data])
    coords_dataset = xr.Dataset(coords={'lon': ('nod2', mesh.x2),
                                        'lat': ('nod2', mesh.y2),
                                        'nz': mesh.zlev,
                                        'nz1': (mesh.zlev[:-1] + mesh.zlev[1:]) / 2.0})

    coords_dataset.coords['lon'].attrs['long_name'] = 'longitude'
    coords_dataset.coords['lon'].attrs['units'] = 'degrees_east'
    coords_dataset.coords['lat'].attrs['long_name'] = 'latitude'
    coords_dataset.coords['lat'].attrs['units'] = 'degrees_north'
    coords_dataset.coords['nz1'].attrs['long_name'] = 'depth at half level'
    coords_dataset.coords['nz1'].attrs['units'] = 'm'
    coords_dataset.coords['nz'].attrs['long_name'] = 'depth'
    coords_dataset.coords['nz'].attrs['units'] = 'm'
    coords_dataset.attrs['Conventions'] = 'CF-1.7'
    return coords_dataset

def open_dataset(path_or_pattern: str, mesh_path: str, abg: Sequence=[50, 15, -90],
                 **kwargs) -> xr.Dataset:
    da = xr.open_mfdataset(path_or_pattern, parallel=True, **kwargs)
    mesh = fesom_mesh_to_xr(mesh_path, *abg)
    return xr.merge([da, mesh])
