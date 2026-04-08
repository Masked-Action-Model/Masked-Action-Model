import numpy as np
from h5py import Dataset, File, Group


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "env_states": "env_states",
    "observations": "obs",
    "success": "success",
    "terminated": "terminated",
    "truncated": "truncated",
    "next_observations": "obs",
    "actions": "actions",
    "mas": "mas",
    "mask": "mask",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    if isinstance(file, Dataset):
        return file[()]
    raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(path):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None, traj_indices=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = sorted(
        [k for k in file.keys() if k.startswith("traj_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if traj_indices is not None:
        selected_keys = []
        for idx in traj_indices:
            idx = int(idx)
            assert 0 <= idx < len(keys), f"traj_idx out of range: {idx} not in [0, {len(keys)})"
            selected_keys.append(keys[idx])
        keys = selected_keys
    elif num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_dataset_meta(path):
    print("Loading HDF5 meta", path)
    file = File(path, "r")
    if "meta" not in file:
        file.close()
        print("Loaded meta: []")
        return {}
    ret = load_content_from_h5_file(file["meta"])
    file.close()
    print("Loaded meta", sorted(ret.keys()))
    return ret


def load_demo_dataset(path, keys=["observations", "actions"], num_traj=None, concat=True, traj_indices=None):
    raw_data = load_traj_hdf5(path, num_traj=num_traj, traj_indices=traj_indices)
    ordered_traj_keys = sorted(raw_data.keys(), key=lambda x: int(x.split("_")[-1]))
    first_key = ordered_traj_keys[0]
    _traj = raw_data[first_key]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in ordered_traj_keys]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)
            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset
