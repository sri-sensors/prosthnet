# About this project

PROSTHNET is a deep learning architecture for prosthetics. 

## Purpose / Vision

PROSTHNET's is a tunable ML encoder for prosthetic sensors. The model is inspired by ganglion cells. The code contains 
a VNE, for 'Virtual Neural Encoder', that map any number of sensor signals in the prosthetic to a fixed number of 
encoded signals used for driving electrodes in an implanted neural interface. The code is used to do nonlinear mappings, 
mix input signal, create new signals from mathematica combinations of inputs, and more. Think of it as sophisticated 
way to control the electroded. The network is very fast, so complex feed-forward operations on the sensors data can be 
implemented in real time. 

Since the VNE can learn, the VNE can be trained to make a prosthetic work better or possibly do new things it did not do 
before. 

The intent of the project is to understand the nature of information flow between physics and biology. AI software helps
us to understand the nature of information flow between these two physical extremes. 

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
Pytorch installation will be specific to your system configuraiton depending on gpu availability and drivers.

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
Documents are available on read-the-docs. You may also create the docs. Configuration files are supplied, simply install 
sphinx,  navigate to ./docs/source, and make the docs
```cmd
pip install sphinx
cd docs
make html
```
(optional) If you wish to modify the code and later merge the code into the master branch, ou will need to update the docs build. 
You'll need to run 
```bash
pip install sphinx
mkdir docs
cd docs
sphinx-quickstart
```
go to document source directory and configure sphinx per the documentation https://www.sphinx-doc.org/en/master/usage/configuration.html

```cmd
cd source
```
edit conf.py, uncomment these lines
```cmd
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

```
in conf.py, add this line to support numpy and Google docstrings or other extensions https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html?highlight=sphinx.ext
```cmd
extensions = ['sphinx.ext.napoleon']
```

generate source files using api-doc (see https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html for details) and return to docs root
```cmd
sphinx-apidoc -o . ../../src/vne
```

finally, build the custom docs
```cmd
cd ..
make clean
```
```cmd
make html
```

### Run tests

Make sure to have completed the development. From the top level directory 'prosthnet' run:

```bash
python -m unittest
```

# Using the library

To use the library, you will first create a network in Pycharm. After that, export the network into a format that is readable 
by whatever software exposes the raw sensor data in the prosthetic. It's then a matter of finding which signals to route into 
and out of the customized encoder. Once the basic wiring is done, then comes the interesting work to optimize the signal flow 
between the prosthetic sensors and the subject wearing the prosthetic. 

To apply the neural network-based encoder to a prosthetic, all you need is the *.onnx file, the associate parameter file, the 
script, and some samples. I created a very lightweight example of how this works at the end of './experiments/Survey_vne.m'. 
No python is required to run the encoder. 

# Matlab 
Exported networks go into the'./src/vne/models' folder. The pycharm neural nets get saved as a *.onnx file. There are 
several ONNX models in the folder already that show different kinds of encoders that can be made by changing the 
properties of the neural network.  

Matlab can read in the *.onnx file, but it needs the deep learning toolbox. Also, the deep learning toolbox will 
need the onnx extension. For more information, read through and test out 'experiments/Survey_vne.m' in the experiments folder. 

# Acknowledgements

This project was sponsored by the AIE-INI program under the Artificial Intelligence Exploration Initiative. 
