import h5py
with h5py.File('./Data/train_patch.hdf5', 'r') as f:
    print(f['chi_cosmos'].shape)
    print(f['phs_tissue'].shape)
    print(f['mask'].shape)