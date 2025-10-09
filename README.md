# RePoseD: Efficient Relative Pose Estimation With Known Depth Information

This is the repository containing the code for ICCV 2025 (oral) paper. DOI: TBA. Pre-print available on Arxiv: [2501.07742](https://arxiv.org/abs/2501.07742)

## Demo

TBA

## Use in your own project

TBA

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

## Extended Results

After finishing the camera-ready version for ICCV we have implemented an improved LO for PoseLib which optimizes the Sampson and reprojection errors jointly. This results in significant improvement in terms of accuracy often surpassing MADPose results while being significantly faster.

Benchmark results are presented [EXTENDED_RESULTS.md](EXTENDED_RESULTS.md). Below is a preview of results on the PhotoTourism dataset.

#### Phototourism (Calibrated)

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">SP+LG</td><td align="center" colspan="3">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>1.42</td><td>76.56</td><td>63.79</td><td>0.78</td><td>86.18</td><td>264.61</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>3P-RelDepth</td><td></td><td></td> <td>8.12</td><td>53.40</td><td>55.85</td><td>1.69</td><td>67.22</td><td>221.06</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>1.40</td><td>77.37</td><td>32.95</td><td>0.78</td><td>86.42</td><td>148.76</td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>1.27</td><td>80.28</td><td>788.18</td><td>0.87</td><td>86.85</td><td>1753.49</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td><strong>1.24</strong></td><td><strong>81.34</strong></td><td><strong>28.93</strong></td><td><strong>0.74</strong></td><td><strong>88.58</strong></td><td><strong>125.66</strong></td>
</tr>
<tr>
<td>Ours*</td><td>✔</td><td></td> <td>1.75</td><td>80.29</td><td>30.11</td><td>1.03</td><td>88.02</td><td>135.95</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>3P-RelDepth</td><td></td><td></td> <td>4.07</td><td>51.60</td><td>52.49</td><td>1.33</td><td>67.56</td><td>214.73</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>1.40</td><td>77.47</td><td>34.30</td><td>0.78</td><td>86.43</td><td>150.95</td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>1.15</td><td>82.09</td><td>720.34</td><td>0.78</td><td>87.60</td><td>1695.57</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td><strong>1.04</strong></td><td>83.71</td><td><strong>30.88</strong></td><td><strong>0.69</strong></td><td>89.27</td><td><strong>131.52</strong></td>
</tr>
<tr>
<td>Ours*</td><td>✔</td><td></td> <td>1.16</td><td><strong>84.56</strong></td><td>31.19</td><td>0.81</td><td><strong>90.18</strong></td><td>137.26</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>1.14</td><td>81.66</td><td>137.75</td>
</tr>
<tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P-RelDepth</td><td></td><td></td> <td><strong>1.13</strong></td><td>80.83</td><td>149.86</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td><strong>1.13</strong></td><td><strong>81.50</strong></td><td><strong>66.06</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>2.10</td><td>72.14</td><td>2154.89</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>29.59</td><td>1.27</td><td>95.77</td>
</tr>
<tr>
<td>Ours*</td><td>✔</td><td></td> <td>29.36</td><td>1.28</td><td>121.02</td>
</tr>
</table>

\* Denotes the use of P3P + our new optimization strategy.


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
