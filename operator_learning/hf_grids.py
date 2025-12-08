import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from joblib import Memory
import os
mem = Memory(location='.cache')

api = HfApi()

def make_config_id(solver, flux, nx, nt, dx, dt, max_steps):
    return f"{solver}_{flux}_nx{nx}_nt{nt}_dx{dx}_dt{dt}_steps{max_steps}"

def upload_grids(grids, solver, flux, nx, nt, dx, dt, max_steps,
                     repo_id="mchami/grids"):

    assert grids.shape[1:] == (nt, nx)
    n_samples = grids.shape[0]

    config_id = make_config_id(solver, flux, nx, nt, dx, dt, max_steps)
    data_path = f"lwr/{config_id}.npz"

    # 1) Save grids locally
    np.savez_compressed("tmp_grids.npz", grids=grids)

    # 2) Upload/overwrite only this file
    api.upload_file(
        path_or_fileobj="tmp_grids.npz",
        path_in_repo=data_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # 3) Update index (small)
    # try to download existing index
    try:
        idx_path = hf_hub_download(repo_id, "index.parquet", repo_type="dataset")
        df = pd.read_parquet(idx_path)
    except Exception:
        df = pd.DataFrame(columns=[
            "config_id", "solver", "flux", "nx", "nt", "dx", "dt",
            "max_steps", "n_samples", "file"
        ])

    # remove old row for this config_id if any
    df = df[df["config_id"] != config_id]

    # add new row
    df = pd.concat([
        df,
        pd.DataFrame([{
            "config_id": config_id,
            "solver": solver,
            "flux": flux,
            "nx": nx,
            "nt": nt,
            "dx": dx,
            "dt": dt,
            "max_steps": max_steps,
            "n_samples": n_samples,
            "file": data_path,
        }])
    ], ignore_index=True)

    # save + upload index
    df.to_parquet("index.parquet")
    api.upload_file(
        path_or_fileobj="index.parquet",
        path_in_repo="index.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # 4) Delete the temporary files
    os.remove("tmp_grids.npz")
    os.remove("index.parquet")

@mem.cache
def download_grids(solver, flux, nx, nt, dx, dt, max_steps, repo_id="mchami/grids"):

    config_id = make_config_id(solver, flux, nx, nt, dx, dt, max_steps)

    # 1) Load index
    idx_path = hf_hub_download(repo_id, "index.parquet", repo_type="dataset")
    df = pd.read_parquet(idx_path)

    row = df[df["config_id"] == config_id]
    if row.empty:
        return None  # config not present on hub

    file_path = row.iloc[0]["file"]
    # 2) Download the NPZ for this config
    npz_local = hf_hub_download(repo_id, file_path, repo_type="dataset")
    grids = np.load(npz_local)["grids"]  # shape (N, nt, nx)

    return grids

if __name__ == "__main__":
    # Test if the upload function works
    grids = np.random.rand(100, 100, 100)
    upload_grids(grids, "solver", "flux", 100, 100, 0.1, 0.1, 3)
    
    # Test if the download function works
    grids = download_grids("solver", "flux", 100, 100, 0.1, 0.1, 3)
    print(f"Downloaded grids shape: {grids.shape}")