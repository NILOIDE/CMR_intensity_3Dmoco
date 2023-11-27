# Intensity-based 3D motion correction for cardiac MR images
This is the code repository for our ISBI submission: 



We propose a method to mitigate the effect of inter-slice motion in CMR images for all SA and LA slices simultaneously by optimizing the 3D rotation and translation parameters on sampled intensities along slice intersections. Our approach is formulated as a subject-specific optimization problem and requires no prior knowledge of the underlying anatomy.

![Alt text](images/4ch_alignment.drawio.png?raw=true "Diagram of short-axis slice alignment")



## Sample results

If two slices are aligned, both slices should have equal intensities along their intersections. We observe clear similarities in intersection intensity patterns (vertical bands) after optimization (green lines) as opposed to the starting alignment (red lines).

![Alt text](images/Sample_result_final.drawio.png?raw=true "Resulting intensity differences after slice alignment optimization")



We sample points along the intersection and minimize the difference in intensities using gradient descent. We implement our algorithm using Pytorch to make use of GPU-acceleration, allowing us to optimize a dataset of slices in under a minute.
 
![Alt text](images/Optimization.png?raw=true "Intensity difference error during optimization")



## Citation and Contribution

Please cite this work if any of our code or ideas are helpful for your research.

```
@inproceedings{stolt2023nisf,
  title={NISF: Neural Implicit Segmentation Functions},
  author={Stolt-Ans{\'o}, Nil and McGinnis, Julian and Pan, Jiazhen and Hammernik, Kerstin and Rueckert, Daniel},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={734--744},
  year={2023},
  organization={Springer}
}
```
