# Extended Results

Since the camera-ready version of the paper we have implemented an improved optimization strategy. Similarly to MADPose, we optimize the Sampson error jointly with both the forward and backward reprojection errors. The implementation of this strategy within PoseLib leads to significantly better resutls compared to those reported in our paper. Our method with the improved strategy surpasses the results of MADPose while being 10-20x faster. Below we provide a sample of these results on the Phototourism dataset for the calibrated case.

## ScanNet

### Calibrated Case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">SP+LG</td><td align="center" colspan="3">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>6.98</td><td>37.95</td><td>50.27</td><td>3.64</td><td>56.18</td><td>209.19</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>3P-RelDepth</td><td></td><td></td> <td>8.68</td><td>33.50</td><td>48.65</td><td>4.01</td><td>52.56</td><td>229.08</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>6.71</td><td>39.23</td><td><strong>26.36</strong></td><td>3.60</td><td>56.57</td><td><strong>116.72</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>5.95</td><td>41.78</td><td>838.82</td><td>3.52</td><td>58.09</td><td>1586.27</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>5.94</td><td>42.05</td><td>37.22</td><td><strong>3.50</strong></td><td>58.47</td><td>204.69</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>5.68</strong></td><td><strong>43.64</strong></td><td>37.76</td><td>3.52</td><td><strong>58.89</strong></td><td>215.19</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>3P-RelDepth</td><td></td><td></td> <td>7.07</td><td>37.74</td><td>74.11</td><td>3.65</td><td>55.95</td><td>363.97</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>6.73</td><td>39.23</td><td><strong>26.49</strong></td><td>3.60</td><td>56.80</td><td><strong>118.75</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>5.93</td><td>41.86</td><td>828.68</td><td>3.49</td><td>58.35</td><td>1590.70</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>5.91</td><td>42.44</td><td>37.07</td><td>3.45</td><td>58.87</td><td>206.21</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>5.34</strong></td><td><strong>44.65</strong></td><td>37.68</td><td><strong>3.43</strong></td><td><strong>59.53</strong></td><td>216.26</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>3.21</td><td>62.88</td><td>163.73</td>
</tr>
<tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P-RelDepth</td><td></td><td></td> <td>3.21</td><td>62.91</td><td>174.26</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>3.21</td><td>62.90</td><td><strong>78.05</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>3.17</strong></td><td><strong>62.99</strong></td><td>2459.38</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>3.19</td><td>62.97</td><td>134.47</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>3.17</strong></td><td>62.73</td><td>155.95</td>
</tr>
</table>

### Unkown and shared focal case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>6-Point</td><td></td><td></td> <td>10.54</td><td>0.14</td><td>28.39</td><td>25.51</td><td>71.04</td><td>4.78</td><td>0.05</td><td>48.67</td><td>47.45</td><td>139.19</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>3P3D</td><td></td><td></td> <td>11.98</td><td>0.15</td><td>26.23</td><td>24.58</td><td><strong>24.05</strong></td><td>5.03</td><td>0.06</td><td>46.15</td><td>44.79</td><td><strong>90.61</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>7.08</td><td>0.09</td><td>36.42</td><td>33.81</td><td>1397.78</td><td>3.92</td><td>0.04</td><td>54.47</td><td>55.43</td><td>2128.23</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>6.54</strong></td><td><strong>0.06</strong></td><td><strong>38.91</strong></td><td><strong>43.25</strong></td><td>34.90</td><td><strong>3.81</strong></td><td><strong>0.03</strong></td><td><strong>56.17</strong></td><td><strong>61.58</strong></td><td>162.36</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>3P3D</td><td></td><td></td> <td>10.48</td><td>0.14</td><td>28.27</td><td>25.34</td><td><strong>24.92</strong></td><td>4.83</td><td>0.05</td><td>48.27</td><td>46.77</td><td><strong>96.78</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>6.90</td><td>0.09</td><td>36.87</td><td>33.71</td><td>1384.13</td><td>3.93</td><td><strong>0.04</strong></td><td>54.76</td><td>56.47</td><td>2081.34</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>6.17</strong></td><td><strong>0.06</strong></td><td><strong>40.62</strong></td><td><strong>43.27</strong></td><td>34.73</td><td><strong>3.80</strong></td><td><strong>0.04</strong></td><td><strong>56.24</strong></td><td><strong>57.71</strong></td><td>164.01</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>6-Point</td><td></td><td></td> <td>3.38</td><td>0.03</td><td>61.09</td><td>63.59</td><td>97.82</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P3D</td><td></td><td></td> <td><strong>3.40</strong></td><td><strong>0.03</strong></td><td>60.29</td><td>62.71</td><td><strong>49.14</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>3.42</td><td><strong>0.03</strong></td><td>59.78</td><td>61.92</td><td>3348.40</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>3.41</td><td><strong>0.03</strong></td><td><strong>60.37</strong></td><td><strong>63.63</strong></td><td>112.36</td>
</tr>
<tr>
<td>MASt3R Opt.</td><td></td><td></td> <td>3.45</td><td>0.06</td><td>59.49</td><td>52.90</td><td>5080.77</td>
</tr>
</table>

### Uknown and different focals case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>7-Point</td><td></td><td></td> <td>17.38</td><td>0.19</td><td>17.90</td><td>18.87</td><td>16.87</td><td>6.75</td><td>0.08</td><td>37.74</td><td>35.25</td><td>60.31</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>4P4D</td><td></td><td></td> <td>16.82</td><td>0.20</td><td>18.12</td><td>18.76</td><td><strong>16.94</strong></td><td>6.54</td><td>0.08</td><td>38.23</td><td>35.95</td><td><strong>66.19</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>10.49</td><td>0.10</td><td>26.54</td><td>31.47</td><td>1763.41</td><td>4.82</td><td>0.05</td><td>47.52</td><td>51.02</td><td>3001.08</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>10.33</strong></td><td><strong>0.08</strong></td><td><strong>28.32</strong></td><td><strong>36.99</strong></td><td>35.20</td><td><strong>4.65</strong></td><td><strong>0.04</strong></td><td><strong>48.57</strong></td><td><strong>57.61</strong></td><td>183.36</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>4P4D</td><td></td><td></td> <td>16.66</td><td>0.19</td><td>17.86</td><td>19.06</td><td><strong>17.09</strong></td><td>6.52</td><td>0.08</td><td>38.43</td><td>36.12</td><td><strong>65.59</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>10.76</td><td>0.10</td><td>27.20</td><td>31.21</td><td>1765.25</td><td>4.77</td><td><strong>0.04</strong></td><td><strong>47.81</strong></td><td>52.12</td><td>2906.99</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>9.52</strong></td><td><strong>0.07</strong></td><td><strong>29.78</strong></td><td><strong>37.90</strong></td><td>35.08</td><td><strong>4.72</strong></td><td><strong>0.04</strong></td><td>47.79</td><td><strong>55.91</strong></td><td>184.65</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>7-Point</td><td></td><td></td> <td>4.01</td><td>0.04</td><td>54.25</td><td>52.79</td><td>37.99</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>4P4D</td><td></td><td></td> <td>4.07</td><td><strong>0.05</strong></td><td>53.41</td><td>50.67</td><td><strong>33.18</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>4.21</td><td><strong>0.05</strong></td><td>52.41</td><td>50.25</td><td>4445.31</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>5.22</td><td><strong>0.05</strong></td><td>46.39</td><td>47.19</td><td>118.59</td>
</tr>
<tr>
<td>MASt3R Opt.</td><td></td><td></td> <td><strong>3.80</strong></td><td><strong>0.05</strong></td><td><strong>56.37</strong></td><td><strong>53.87</strong></td><td>5080.77</td>
</tr>
</table>


## PhotoTourism

### Calibrated Case

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
<td>Ours</td><td>✔</td><td></td> <td>1.75</td><td>80.29</td><td>30.11</td><td>1.03</td><td>88.02</td><td>135.95</td>
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
<td>Ours</td><td>✔</td><td></td> <td>1.16</td><td><strong>84.56</strong></td><td>31.19</td><td>0.81</td><td><strong>90.18</strong></td><td>137.26</td>
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
<td>Ours</td><td>✔</td><td></td> <td>29.36</td><td>1.28</td><td>121.02</td>
</tr>
</table>

### Uknown and different focals case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>7-Point</td><td></td><td></td> <td>8.03</td><td>0.17</td><td>38.14</td><td>23.78</td><td>24.80</td><td>4.30</td><td>0.10</td><td>53.16</td><td>34.73</td><td>75.10</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>4P4D</td><td></td><td></td> <td>7.71</td><td>0.16</td><td>39.12</td><td>24.44</td><td>22.82</td><td>4.22</td><td>0.10</td><td>53.61</td><td>34.99</td><td>77.14</td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>3.88</strong></td><td><strong>0.07</strong></td><td><strong>57.21</strong></td><td><strong>42.80</strong></td><td>1953.23</td><td><strong>2.41</strong></td><td><strong>0.05</strong></td><td><strong>69.08</strong></td><td><strong>52.39</strong></td><td>3663.43</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>6.10</td><td>0.08</td><td>47.21</td><td>37.59</td><td><strong>14.97</strong></td><td>3.02</td><td><strong>0.05</strong></td><td>64.78</td><td>50.94</td><td><strong>73.75</strong></td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>4P4D</td><td></td><td></td> <td>7.67</td><td>0.16</td><td>39.18</td><td>24.61</td><td><strong>23.63</strong></td><td>4.20</td><td>0.10</td><td>53.84</td><td>35.24</td><td><strong>77.78</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>3.38</td><td>0.06</td><td>60.15</td><td>47.02</td><td>1945.50</td><td>2.30</td><td><strong>0.04</strong></td><td>69.95</td><td>53.90</td><td>3440.95</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td><strong>3.29</strong></td><td><strong>0.05</strong></td><td><strong>60.83</strong></td><td><strong>51.25</strong></td><td>26.14</td><td><strong>1.98</strong></td><td><strong>0.04</strong></td><td><strong>72.68</strong></td><td><strong>59.43</strong></td><td>105.02</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>7-Point</td><td></td><td></td> <td>4.39</td><td>0.09</td><td>54.01</td><td>35.02</td><td>39.53</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>4P4D</td><td></td><td></td> <td>5.20</td><td>0.11</td><td>49.69</td><td>31.64</td><td><strong>36.90</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>10.11</td><td>0.24</td><td>35.16</td><td>20.56</td><td>4871.24</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>33.62</td><td>0.83</td><td>1.17</td><td>2.73</td><td>77.14</td>
</tr>
<tr>
<td>MASt3R Opt.</td><td></td><td></td> <td><strong>2.71</strong></td><td><strong>0.04</strong></td><td><strong>66.54</strong></td><td><strong>56.43</strong></td><td>4903.10</td>
</tr>
</table>


## ETH3D

### Calibrated case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">SP+LG</td><td align="center" colspan="3">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>0.91</td><td>87.67</td><td>48.14</td><td>0.56</td><td>91.10</td><td>184.36</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>3P-RelDepth</td><td></td><td></td> <td>4.74</td><td>72.08</td><td>42.19</td><td>2.74</td><td>82.04</td><td>170.29</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>0.91</td><td>87.67</td><td><strong>25.72</strong></td><td>0.54</td><td>91.16</td><td><strong>111.74</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>0.86</strong></td><td><strong>88.26</strong></td><td>566.16</td><td>0.50</td><td>91.17</td><td>1414.96</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td><strong>0.86</strong></td><td>87.81</td><td>31.39</td><td>0.49</td><td>91.31</td><td>129.21</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>0.94</td><td>87.75</td><td>32.20</td><td><strong>0.46</strong></td><td><strong>91.32</strong></td><td>130.06</td>
</tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>3P-RelDepth</td><td></td><td></td> <td>1.36</td><td>78.82</td><td>49.70</td><td>0.70</td><td>88.25</td><td>207.86</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td>0.88</td><td>88.00</td><td><strong>25.93</strong></td><td>0.56</td><td>91.11</td><td><strong>112.65</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>0.86</strong></td><td><strong>88.03</strong></td><td>558.61</td><td>0.53</td><td><strong>91.33</strong></td><td>1402.46</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>0.88</td><td>87.59</td><td>32.78</td><td><strong>0.50</strong></td><td>91.31</td><td>135.72</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>0.91</td><td>87.37</td><td>32.87</td><td>0.53</td><td>91.14</td><td>136.09</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="3" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>mAA $\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>5-Point</td><td></td><td></td> <td>0.66</td><td>90.29</td><td>126.77</td>
</tr>
<tr>
<td rowspan="5" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P-RelDepth</td><td></td><td></td> <td><strong>0.67</strong></td><td><strong>90.30</strong></td><td>104.46</td>
</tr>
<tr>
<td>P3P</td><td></td><td></td> <td><strong>0.67</strong></td><td>90.24</td><td><strong>56.52</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>0.92</td><td>87.96</td><td>2647.02</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td>✔</td> <td>0.82</td><td>89.09</td><td>81.64</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>0.85</td><td>87.73</td><td>87.17</td>
</tr>
</table>

### Uknown and shared focals case

<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">SP+LG</td><td align="center" colspan="5">RoMA</td></tr>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>6-Point</td><td></td><td></td> <td>2.45</td><td>0.04</td><td>75.57</td><td>61.52</td><td>80.02</td><td>1.15</td><td>0.02</td><td>85.22</td><td>75.03</td><td>89.48</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">MoGe</td>
<td>3P3D</td><td></td><td></td> <td>3.18</td><td>0.06</td><td>68.24</td><td>54.64</td><td><strong>27.70</strong></td><td>1.60</td><td><strong>0.02</strong></td><td>81.02</td><td>70.26</td><td><strong>60.37</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>1.50</strong></td><td><strong>0.03</strong></td><td>79.23</td><td>66.34</td><td>956.33</td><td><strong>0.89</strong></td><td><strong>0.02</strong></td><td>86.99</td><td>76.66</td><td>1923.31</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>1.55</td><td><strong>0.03</strong></td><td><strong>79.78</strong></td><td><strong>68.93</strong></td><td>30.18</td><td><strong>0.89</strong></td><td><strong>0.02</strong></td><td><strong>87.65</strong></td><td><strong>78.72</strong></td><td>94.10</td>
</tr>
<td rowspan="3" style="vertical-align : middle;text-align:center;">UniDepth</td>
<td>3P3D</td><td></td><td></td> <td>3.49</td><td>0.07</td><td>69.47</td><td>55.57</td><td><strong>27.57</strong></td><td>1.86</td><td><strong>0.02</strong></td><td>82.40</td><td>71.80</td><td><strong>60.01</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td><strong>1.27</strong></td><td><strong>0.03</strong></td><td><strong>81.68</strong></td><td><strong>69.64</strong></td><td>1107.04</td><td>0.83</td><td><strong>0.02</strong></td><td><strong>87.30</strong></td><td><strong>77.34</strong></td><td>1997.88</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>1.46</td><td><strong>0.03</strong></td><td>81.48</td><td>67.08</td><td>31.00</td><td><strong>0.82</strong></td><td><strong>0.02</strong></td><td>87.05</td><td>76.26</td><td>102.72</td>
</tr>
</table>
<table>
<tr><td rowspan="2"  style="vertical-align : middle;text-align:center;">Depth</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Method</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Scale</td><td rowspan="2"  style="vertical-align : middle;text-align:center;">Shift</td><td colspan="5" align="center">MASt3R</td>
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f\uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>6-Point</td><td></td><td></td> <td>1.23</td><td>0.03</td><td>82.99</td><td>69.00</td><td>85.89</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P3D</td><td></td><td></td> <td>1.37</td><td>0.03</td><td>81.30</td><td>67.31</td><td><strong>38.96</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>2.43</td><td>0.05</td><td>72.28</td><td>58.32</td><td>3825.11</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>2.87</td><td>0.05</td><td>70.25</td><td>59.30</td><td>67.01</td>
</tr>
<tr>
<td>MASt3R Opt.</td><td></td><td></td> <td><strong>1.32</strong></td><td><strong>0.01</strong></td><td><strong>85.64</strong></td><td><strong>82.95</strong></td><td>4800.37</td>
</tr>
</table>




