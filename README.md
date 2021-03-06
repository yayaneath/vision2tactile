# vision2tactile: Feeling Touch by Sight

This repo holds 3D point clouds of objects that were grasped using [GeoGrasp](https://github.com/yayaneath/GeoGrasp) and the corresponding tactile data generated by two BioTac SP tactile sensors. The sensors were installed on the middle finger and the thumb of a Shadow Dexterous hand, which was mounted on a Mitsubishi PA10 robotic arm.

There are 9 objects from the [YCB object set](http://www.ycbbenchmarks.com/) plus 3 extra objects: a crisps can, a coffe can, a bleach bottle, a soup can, a snaks box, a sugar box, a meat can, a wood log, a toy train, a toy drill, a stuffed rugby ball and a show. Inside of each folder, there is a unique folder for each sample. In total, we have recorded 600 real robotic grasps: there are 50 samples per object, being the training set the 400 samples corresponding to the first 8 objects and the testing set is composed of the 200 samples generated with the last 4 objects. Each sample has the following files:

- **best-grasp.txt:** a text file with the 3D coordinates of the two grasping points used for executing the contact action.
- **cloud-object.pcd:** a PCD file with the 3D point cloud of the object segmented. These point clouds were recorded using the Intel RealSense D415 camera.
- **tactile.tac:** a text file with the tactile data registered by the BioTac SP sensors. They are organised as follows: firstly, the values of the 24 electrodes of the sensor on the middle finger, followed by its PAC0, PAC1, PDC, TAC and TDC values. Secondly, these same values but for the thumb's sensor.

### Robotic setup

<img src="grasping.jpeg" width="500">

### Objects

<img src="objects.png" width="600">

# Citation
```
@INPROCEEDINGS{vision2tactile,
  author = {Zapata-impata, Brayan S. and Gil, Pablo and Torres, Fernando},
  conference = {Robotics: Science and Systems (RSS 2019). Workshop on Closing the Reality Gap in Sim2real Transfer for Robotic Manipulation},
  title = {{vision2tactile: Feeling Touch by Sight}},
  url = {https://sim2real.github.io/assets/papers/zapata.pdf},
  year = {2019}
}
