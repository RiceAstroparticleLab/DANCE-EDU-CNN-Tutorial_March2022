# DANCE-EDU workshop 2022 @ Rice: A tutorial to CNNs

In this repository you can find a small tutorial notebook about CNNs in python and tensorflow. The goal of the notebook is to provide you with a hand on example such that you get some basic knowledge about the idea behind CNNs. In addition, it provides you some information about the code required to build you own custom CNN. 

The example problem I am using in the tutorial is related to the water Cherenkov neutron-veto of XENONnT. In the example we will try to distinguish neutron-capture signals from background signals by using spatial and time information of the recorded hit-patterns.

The code in this repository is structure as follows:
* The simulator directory contains some code which allows you to generate some toy Events for the above-described problem.
* tools.py contains a few smaller helper functions which will be used during the tutorial. 
* CNNTutorial.ipynb is the tutorial notebook itself. It comes with examples and a lot of explanations. 

