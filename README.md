# RePoseD: Efficient Relative Pose Estimation With Known Depth Information

<a href="https://kocurvik.github.io/reposed/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

This is the repository containing the code for ICCV 2025 (oral) paper. DOI: TBA. Pre-print available on Arxiv: [2501.07742](https://arxiv.org/abs/2501.07742)

## Demos

We provide a [Google Colab demo](demo/reposed_demo.ipynb) for the estimation of relative pose of two images and following dense two-view reconstruction.

<a href="https://colab.research.google.com/github/kocurvik/mdrp/blob/main/demo/reposed_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We also provide a [script](make_pair.py) to create a nice visualization of the resulting pointcloud.

You can also try a [script](make_video.py) to perform dynamic scene reconstruction. Note that this script does not currently include foreground/background segmentation so it works only for videos with static background with sufficient features for matching.

To run the last two scripts you need to install [MoGe](https://github.com/microsoft/MoGe), [LightGlue](https://github.com/cvg/LightGlue) and [Open3D](https://www.open3d.org). You will also need to follow the instructions in the next section.

## Use in your own project

To use RePoseD in your own project you must first install [PoseLib with our PR](https://github.com/PoseLib/PoseLib/pull/152).

```shell
pip install git+https://github.com/kocurvik/PoseLib@pr-mdrp
```

If this is not sufficient you may need to first install some extra packages and/or clone the repo manually:
```shell
git clone https://github.com/kocurvik/PoseLib
cd PoseLib
git checkout pr-mdrp
pip install pybind11_stubs
apt-get install libeigen3-dev
python setup.py install
```

Once installed you can use the new methods added to poselib for relative pose estimation.

```python
import poselib

# extract keypoints and their corresponding depths
# make sure you remove any nans or infs

# set your thresholds
ransac_dict = {'max_epipolar_error': 2.0, 'max_reproj_error': 16.0}
# set this to true if you also want to estimate shit (calib case only)
ransac_dict['estimate_shift'] = False

# use this loss for better estimation in final optimization
bundle_dict = {'loss_type': 'TRUNCATED_CAUCHY'}

# if you know intrinsics you can use this
camera1 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f1, px1, py1]}
camera2 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f2, px2, py2]}
pose, info = poselib.estimate_monodepth_pose(points1, points2, depths1, depths2, camera1, camera2, ransac_dict, bundle_dict)

# for uknown and shared focals you can use (pp is the principal point - usually image center)
image_pair, info = poselib.estimate_monodepth_shared_focal_pose(points1 - pp1, points2 - pp2, depths1, depths2, ransac_dict, bundle_dict)
f = image_pair.camera1.focal()
pose = image_pair.pose

# for uknown and different focals you can use
image_pair, info = poselib.estimate_monodepth_varying_focal_pose(points1 - pp1, points2 - pp2, depths1, depths2, ransac_dict, bundle_dict)
f1 = image_pair.camera1.focal()
f2 = image_pair.camera2.focal()
pose = image_pair.pose

# to transform the pointcloud from the first image into the coordinates of the second image you can use:
xyz1_in_camera2_frame = (1/pose.scale) * ((pose.R @ xyz1.T).T + pose.t)
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
</table>

\* Denotes the use of P3P + our new optimization strategy.


## ICCV (2025) Evaluation

To run the experiments from the paper you can use the provided evaluation code. We are currently working on improvements to the methods. For reproducibility we keep the original code available in the `iccv-eval` branches.

### Setting up eval repo, PoseLib and Madpose

You will need to clone this repo and install our PoseLib and Madpose forks with all variants.

Note that the PoseLib variant used in the evaluation scripts is different from the one mentioned in previous sections which includes only our version.

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
