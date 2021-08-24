# About this project

PROSTHNET is a deep learning architecture for prosthetics. It is applied to understand and engineer information flow between 
physical and biological system.  

## Purpose / Vision

PROSTHNET is a tunable machine learnable encoder for prosthetic sensors inspired by ganglion cells. The code contains a VNE for 'Virtual Neural Encoder' that maps any sensor signals in the prosthetic to a fixed number of encoded signals used for driving electrodes in an implanted neural interface. The VNE can apply nonlinear mappings, 
mix input signals, create new signals from mathematical combinations of inputs, and more. Think of it as a sophisticated way to control the electrodes. The network is fast and enables feed-forward operations on sensor data can be implemented in real-time. 

The machine learnability of the VNE can be applied to condition the encoder output with new data sources or non-biomimetic sensors.

# Content

* Readme.md template (this document)
* source code inside 'src/vne'
* test code inside 'test'
* documentation 'docs'
* python module definition template
  - setup.cfg
  - MANIFEST.in the manifest allows adding non-python module specific components to the delivery (e.g. documentation, changelog, network parameters, etc.)
* general meta documents\
  - LICENSE
* docker configuration (build and docker-compose)
* .gitignore

## System requirements

This project requires:

* `Python 3.8` or newer
* `Pytorch 1.8` or newer

## Installation

This section describes installing this project as a python module.

## Installation

#### Install Dependencies

Install Pytorch from the [Pytorch page](https://pytorch.org/get-started/locally/)
Pytorch installation will be specific to your system configuration depending on GPU availability and drivers.

#### Install from PyPi
This method does not currently work but should be working soon.
```bash
pip install prosthnet
```

#### Install from source
This is currently the preferred method.
```bash
git clone https://github.com/brillouinzone/prosthnet.git
```

```cmd
cd prosthnet
```
For end user installation:
```cmd
pip install .
```

### Build docs
Documents are available in the ./docs folder. You may also create the docs. Configuration files are supplied; simply install 
sphinx,  navigate to ./docs/source, and type the commands
```cmd
pip install sphinx
cd docs
make html
```

### Run tests

Make sure to have completed the development. From the top-level directory 'prosthnet' run:

```bash
python -m unittest
```

# Using the library

To use the library, create a deep network in PyTorch (./src/model/). After that, export the encoder network to an ONNX format. 
Then import the saved ONNX network into MATLAB or whatever software interfaces to the raw sensor data in the prosthetic. 

After finishing the signal flow, the exciting work is to optimize the *information* flow between the prosthetic sensors and the subject wearing the prosthetic. 

# Matlab 
Exported networks appear in the'./src/vne/models' folder. The PyTorch neural nets get saved as a *.onnx file. 

MATLAB uses the deep learning toolbox to read in the *.onnx file. The MATLAB script 'experiments/Survey_vne.m' proves a step by step walkthrough of interfacing MATLAB to PyTorch.

# Acknowledgements

This project was sponsored by the AIE-INI program under the Artificial Intelligence Exploration Initiative. 