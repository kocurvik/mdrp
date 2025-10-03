# RePoseD: Efficient Relative Pose Estimation With Known Depth Information

This is the repository containing the code for ICCV 2025 (oral) paper. DOI: TBA. Pre-print available on Arxiv: [2501.07742](https://arxiv.org/abs/2501.07742)

## Demo

TBA

## Use in your own project

TBA

## ICCV (2025) Evaluation

To run the experiments from the paper you can use the provided evaluation code. We are currently working on improvements to the methods. For reproducibility we keep the original code available in the `iccv-eval` branches.

### Setting up eval repo, PoseLib and Madpose

You will need to clone this repo and install our PoseLib and Madpose forks with all variants.

```shell
# create a conda environment
conda create -n mdrp
conda activate mdrp
conda install numpy, scipy, tqdm, hd5py, tectonic, prettytable, matplotlib, seaborn, eigen=3.4

# cloning this repo
git clone -b iccv-eval https://github.com/kocurvik/mdrp

# installing Madpose
git clone --recursive https://github.com/kocurvik/madpose
cd madpose
# this step is optional if you have the libs installed
# if you do not have root access you can install them for user or using conan etc.
# sudo apt-get install libceres-dev libopencv-dev
pip install . -v
cd ..

# installing PoseLib fork with all evaluated variants
# note that you need to have Eigen version 3.4 to run this
git clone -b iccv-eval --recursive https://github.com/kocurvik/PoseLib-mdrp
cd PoseLib-mdrp
pip install . -v
cd ..
```

### Download the eval dataset

You can download the extracted matches along with dephts obtained using various MDE methods used in our paper (link TBA). 

### Running the evaluation scripts

You can run the evaluations using the eval scripts. For each scene you can run:
```shell
conda activate mdrp

cd mdrp
export PYTHONPATH=.

# -nw parameter is based on the number of CPUs available
# replace e.g. /path/to/dataset/features/dataset/scene -> /path/to/dataset/splg/pt/florence_cathedral_splg.h5 
python eval.py -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5
python eval_shared_f.py -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5
python eval_varying_f.py -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5

# this runs the graph-based evals - not all results are included in paper
python eval.py -g -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5
python eval_shared_f.py -g -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5
python eval_varying_f.py -g -t 2.0 -r 16.0 -a -o -nw 16 /path/to/dataset/per/individual/scene.h5

```


We used a SLURM-based cluster to run the jobs. You can run experiments using the script to spawn jobs for each scene for all experiments.

```shell
# Modify the slurm script before running to have correct paths etc.
# Here you should run it for the desired dataset type e.g. splg/pt/ for SP+LG and Phototourism, roma/eth3d for RoMA on ETH3D etc...
# The last number is used just for logging.
/path/to/mdrp/slurm_scripts/eval_mdrp_spawn_all.sh /path/to/one/of/dataset/mdrp/splg/pt/ 001 
```

### Generating tables and graph

You can generate all the tables from the paper using the following scripts
```shell
cd mdrp
export PYTHONPATH=.
# Main paper tables
python tables.py
# Main paper graphs
python vis.py
# SM tables
python tables_sideways.py
```
## Citation

If you find our work useful please consider citing:

```
@inproceedings{ding2025reposed,
  title={RePoseD: Efficient Relative Pose Estimation With Known Depth Information},
  author={Ding, Yaqing and Kocur, Viktor and V{\'a}vra, V{\'a}clav and Haladov{\'a}, Zuzana Berger and Yang, Jian and Sattler, Torsten and Kukelova, Zuzana},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
