# DANCE-EDU workshop 2022 @ Rice: A tutorial to CNNs

[![DOI](https://zenodo.org/badge/478660047.svg)](https://zenodo.org/badge/latestdoi/478660047)

In this repository you can find a small tutorial notebook about CNNs using python and tensorflow. The goal of the notebook is to provide you with a hand on example such that you get some basic knowledge about the idea behind CNNs. In addition, it provides you some information about the code required to build you own custom CNN. 

The tutorial assumes that you have already some prior knowledge in python and packages which are typically used in the scientific context like numpy and matplotlib. In addition, I assume that you are familiar with git and you know how to download the project. If not you can find some instructions [here](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

The example problem I am using in the tutorial is related to the water Cherenkov neutron-veto of XENONnT. In the example we will try to distinguish neutron-capture signals from background signals by using spatial and time information of the recorded hit-patterns.
The code in this repository is structure as follows:

* The simulator directory contains some code which allows you to generate some toy Events for the above-described problem.
* tools.py contains a few smaller helper functions which will be used during the tutorial. 
* CNNTutorial.ipynb is the tutorial notebook itself. It comes with examples and a lot of explanations.
To install all the requirements for the tutorial please navigate into the project directory and run to install all required packages. 
```
pip install -r requirements.txt
```
Afterwards you can begin the tutorial by starting `CNNTutorial.ipynb`.

