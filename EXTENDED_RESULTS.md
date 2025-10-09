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
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f \uparrow$</td><td>Runtime (ms)</td><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f \uparrow$</td><td>Runtime (ms)</td></tr>
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
<tr><td>$\epsilon(^\circ)\downarrow$</td><td>$\xi\downarrow$</td><td>mAA $\uparrow$</td><td>mAA$_f \uparrow$</td><td>Runtime (ms)</td></tr>
<td rowspan="1" style="vertical-align : middle;text-align:center;">-</td>
<td>6-Point</td><td></td><td></td> <td>3.38</td><td>61.09</td><td>97.82</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align : middle;text-align:center;">MASt3R</td>
<td>3P3D</td><td></td><td></td> <td><strong>3.40</strong></td><td>60.29</td><td><strong>49.14</strong></td>
</tr>
<tr>
<td>MADPose</td><td>✔</td><td>✔</td> <td>3.42</td><td>59.78</td><td>3348.40</td>
</tr>
<tr>
<td>Ours</td><td>✔</td><td></td> <td>3.41</td><td><strong>60.37</strong></td><td>112.36</td>
</tr>
<tr>
<td>MASt3R Opt.</td><td></td><td></td> <td>3.45</td><td>59.49</td><td>5080.77</td>
</tr>
</table>

