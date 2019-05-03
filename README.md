# MMCBM_Multiple_Model_Consistency_Based_Model
Object classification model leveraging deep convolutional features and manifold processing
For details theory read the paper titled Neighbourhood Consistency Based Deep Domain Adaption Analysis for Multi Category Object Detection [1]. Performance of the model are depicted bellow.

![Accuracy Result](/Images/Table1.png)

![F1_Score](/Images/Table2.png)

The resultant feature subset elucidates lower sum of intracluster distances of samples compared to the full set of features.

To execute the proram:
*.Setup libSVM (Also setup C Compilers if required)
*.Setup all variables in initData.m and run in Matlab.

## Dependencies
* Matlab/Octave
* libSVM

## Reference
[1]. B. Pal and B. Ahmed, "Neighbourhood consistency based deep domain adaption analysis for multi category object detection," 2016 19th International Conference on Computer and Information Technology (ICCIT), Dhaka, 2016, pp. 395-398. doi: 10.1109/ICCITECHN.2016.7860230
