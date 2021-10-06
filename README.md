![alt text](qualitative.png?raw=true)

This repository contains the necessary code to run the methodology described in:<br>
https://arxiv.org/abs/2110.01614

<h3>SIMULATION</h3>
To run a cloth simulation use:<br>
python run.py [GPU_ID] [NAME]<br>
where GPU_ID defines which GPU/s to use and NAME is the name of the files where the results will be stored.<br>
Results will be stored as:<br>
'results/[NAME].obj' : a 3D mesh for the initial cloth state<br>
'results/[NAME].pc2' : animation data for the cloth mesh in PC2 format<br>
These results should be easily viewed in Blender. <br>
NOTE: if using Blender, make sure to set 'Keep Verts Order' when importing the OBJ file.<br>
<br>
You will find many simulation parameters at the beginning of the script 'run.py'.<br>
You will find parameters related to the Neural Collider in the script 'run.py', lines 83-89.<br>
The code, as it is, is ready to run simulators using two colliders, the stanford bunny and dragon. Checkpoints are provided.<br>
Nonetheless, it should be easily scalable to new objects.

<h3>TRAINING</h3>
Within the folder 'SDF/' you will find all the code needed to train new SDF.<br>
To train a model:<br>
python train.py [GPU_ID] [NAME]<br>
where GPU_ID is the GPU to use and NAME is the name for checkpoint file.<br>
Set 3D object data in 'train.py' line 80-81.<br>
Training data can be provided as OBJ or NPY. More info below, see 'Data'.

<h4>Checkpoints</h4>
In this folder you will find checkpoints for the bunny and dragon models.<br>
The functionality to save and load checkpoints is written within the model.

<h4>Data</h4>
It contains the data pipeline implementation.<br>
You can provide the Data object (defined in 'data.py') with an OBJ file or a NPY file. This is done in 'train.py', lines 82-83.<br>
If OBJ, it will sample points and compute distances on the fly.<br>
If NPY, it expects a pre-computed Nx4 matrix with 'xyz' and signed distance 'd' as (x,y,z,d) for each point.<br>
The script 'data.py' can be used to pre-compute the points. Check lines 91-108.<br>
Pre-computing the points heavily speeds up the training.

<h4>Model</h4>
Model implementation.<br>
There are specific implementations for the stanford bunny and dragon. Their architecture corresponds to the provided checkpoints.<br>
The simulation script already chooses the correct model for the bunny and dragon.<br>
There is a generic model implementation 'SDF.py'. Training script will import this model.<br>
The architecture can be easily defined in 'SDF.py', function '_build(self)', line 18.<br>
Layers implementation found in 'Layers.py'.

<h4>Objects</h4>
Put your OBJ files in this folder. It will also expect NPY files in this folder. 'data.py' automatically places NPY in this folder.<br>
Currently it is empty due to size.

