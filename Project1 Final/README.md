# DeepProj
Deep Learning Project 1 EPFL 2019

Tomas Giro, Daniele Rocchi, Giulio Trichilo

--------------------------

This folder contains the relevant files fror the first mini project. The structure is as follows:

.
├── data
│   └── mnist
├── Project1_Final.ipynb
├── README.md
└── Utils
    ├── DataImport.py
    ├── dlc_practical_prologue.py
    ├── errs.py
    ├── __init__.py
    ├── Networks.py

Data: contains MNIST downloaded data

Project_1_Final: Jupyter notebook with model training and output for network. Uses two loss functions (commented in code).

Utils: Folder with function for error calculation, the pytorch network, and a DataImporter class for importing data conveniently.

	Errs: calculate number of errors, can be used in train and test based on inputs
	DataImport: Imports data and puts it into a class
	Networks.py: Contains ONE class CNN_SP which implements 2 convolutional layers and 2 fc layers, all with weight sharing, and then one final layer 20x2 for classification.





