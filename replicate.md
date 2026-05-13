## 下载HOI4D video数据集
https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8xMmU1YzNkYmVmZmQwNTk0L0VaUUZfZV9idy1VZ2dCSXZBUUFBQUFBQjRBY0RPeGpfdXVoN2FsUmFSOWI3TVE%5FZT1HeUJOYUQ&cid=12E5C3DBEFFD0594&id=12E5C3DBEFFD0594%21303&parId=12E5C3DBEFFD0594%21283&o=OneUp

## 下载Ego4D-video 20k subset数据集
https://huggingface.co/datasets/weikaih/ego4d-random-views-20k/tree/main/data


## 配置EgoSim dataset process pipeline
REPOS_DIR="./repos"   # e.g. ~/repos
mkdir -p "${REPOS_DIR}" && cd "${REPOS_DIR}"

# Depth-Anything-3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git

# SAM3 (Segment Anything Model 3)
git clone https://github.com/facebookresearch/sam3.git

# HaMeR (with ViTPose submodule)
git clone --recursive https://github.com/geopavlakos/hamer.git

modelscope download --model facebook/sam3 sam3.pt --local_dir ${REPOS_DIR}/sam3/checkpoints
