## Neural Implicit Surfaces for Efficient and Accurate Collisions in Physically Based Simulations

![alt text](qualitative.png?raw=true)

<a href="https://hbertiche.github.io/NeuralColliders">Project Page</a> | <a href="https://arxiv.org/abs/2110.01614">arXiv</a> | <a href="https://youtu.be/F1kQrYXWtEI">Video</a>

## Abstract
>
>
>Current trends in the computer graphics community propose leveraging the massive parallel computational power of GPUs to accelerate physically based simulations. Collision detection and solving is a fundamental part of this process. It is also the most significant bottleneck on physically based simulations and it easily becomes intractable as the number of vertices in the scene increases. <i>Brute force</i> approaches carry a quadratic growth in both computational time and memory footprint. While their parallelization is trivial in GPUs, their complexity discourages from using such approaches. Acceleration structures &mdash;such as BVH&mdash; are often applied to increase performance, achieving logarithmic computational times for individual point queries. Nonetheless, their memory footprint also grows rapidly and their parallelization in a GPU is problematic due to their branching nature. We propose using implicit surface representations learnt through deep learning for collision handling in physically based simulations. Our proposed architecture has a complexity of $\mathcal{O}(n)$ &mdash;or $\mathcal{O}(1)$ for a single point query&mdash; and has no parallelization issues. We will show how this permits accurate and efficient collision handling in physically based simulations, more specifically, for cloth. In our experiments, we query up to 1M points in $\sim300$ milliseconds.

<a href="mailto:hugo_bertiche@hotmail.com">Hugo Bertiche</a>, <a href="mailto:mmadadi@cvc.uab.cat">Meysam Madadi</a> and <a href="https://sergioescalera.com/">Sergio Escalera</a>

## Simulation

To run a cloth simulation use:
```
python run.py [GPU_ID] [NAME]
```
where GPU_ID defines which GPU/s to use and NAME is the name of the files where the results will be stored.<br>
Results will be stored as:<br>
```results/[NAME].obj``` : a 3D mesh for the initial cloth state<br>
```results/[NAME].pc2``` : animation data for the cloth mesh in PC2 format<br>
These results should be easily viewed in Blender. <br>
NOTE: if using Blender, make sure to set ```Keep Verts Order``` when importing the OBJ file.<br>
<br>
You will find many simulation parameters at the beginning of the script 'run.py'.<br>
You will find parameters related to the Neural Collider in the script 'run.py', lines 83-89.<br>
The code, as it is, is ready to run simulators using two colliders, the stanford bunny and dragon. Checkpoints are provided.<br>
Nonetheless, it should be easily scalable to new objects.

## Training

Within the folder ```SDF/``` you will find all the code needed to train new SDF.<br>
To train a model:
```
python train.py [GPU_ID] [NAME]
```
where GPU_ID is the GPU to use and NAME is the name for checkpoint file.<br>
Set 3D object data in ```train.py``` line 80-81.<br>
Training data can be provided as OBJ or NPY. More info below, see 'Data'.

## Checkpoints

In this folder you will find checkpoints for the bunny and dragon models.<br>
The functionality to save and load checkpoints is written within the model.

## Data

It contains the data pipeline implementation.<br>
You can provide the Data object (defined in ```data.py```) with an OBJ file or a NPY file. This is done in ```train.py```, lines 82-83.<br>
If OBJ, it will sample points and compute distances on the fly.<br>
If NPY, it expects a pre-computed Nx4 matrix with ```xyz``` and signed distance ```d``` as ```(x,y,z,d)``` for each point.<br>
The script ```data.py``` can be used to pre-compute the points. Check lines 91-108.<br>
Pre-computing the points heavily speeds up the training.

## Model

Model implementation.<br>
There are specific implementations for the stanford bunny and dragon. Their architecture corresponds to the provided checkpoints.<br>
The simulation script already chooses the correct model for the bunny and dragon.<br>
There is a generic model implementation ```SDF.py```. Training script will import this model.<br>
The architecture can be easily defined in ```SDF.py```, function ```_build(self)```, line 18.<br>
Layers implementation found in ```Layers.py```.

## Objects

Put your OBJ files in this folder. It will also expect NPY files in this folder. ```data.py``` automatically places NPY in this folder.<br>
Currently it is empty due to size.

## Citation
```
@article{DBLP:journals/corr/abs-2110-01614,
  author = {Hugo Bertiche and Meysam Madadi and Sergio Escalera},
  title = {Neural Implicit Surfaces for Efficient and Accurate Collisions in Physically Based Simulations},
  journal = {CoRR},
  volume = {abs/2110.01614},
  year = {2021},
  url = {https://arxiv.org/abs/2110.01614},
  eprinttype = {arXiv},
  eprint = {2110.01614},
  timestamp = {Fri, 08 Oct 2021 15:47:55 +0200},
  biburl = {https://dblp.org/rec/journals/corr/abs-2110-01614.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
