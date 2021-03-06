%% Copyright 2021 SRI International -- Virtual Neural Encoder (VNE)
% 
%  external signal (physics) -> encoder (signal processing) -> neuroelectric stimulation (amplification) -> neural spikes (biology and perception)
%
%  Signal encoders condition external data prior to a
%  neuroelectric interface with the goal of optimizing the information
%  flow between the outside world and the representation of that signal inside 
%  a neuron. 
%  
%  The VNE is an encoder that uses deep learning to optimize the properties
%  of the encoder. This script reads in a saved VNE model and creates a
%  function that performs nonlinear filterning of input data in real time. 
%
%  This file contains a few simple, hand-tuned examples of
%  encoders. It also possible to train the network with standard ML methods.
% 
%  The goal of this script is to demonstrates a basic interaction between 
%  pytorch and Matlab. Networks were created in pytorch and exported to a file 
%  in .src/vne/experiments/vneTest.py.
%  
%  The exported network file uses the ONNX file format, which is a portable way
%  of representing a neural network. 
%
%  This script shows how to open the ONNX file, run realtime samples through 
%  the network, and examine the output. 
%
%  Several network models stored in ./src/vne/models/ were used to alter
%  the property of the encoder. As it shows, the behavior of the encoder
%  can be very simple or very complex. 
%
% 
%  8/20/2021 - Initial version David Stoker
%  8/24/2021 - Add 19 sensor network
%  http://github.com/brillouinzone/ini-prosthnet
%% Requirements
% 
%  Matlab 
%  Matlab-Deep Learning Toolbox
%  Matlab ONNX plugin (click yes when asked to install this plugin)
% 
%% Load the Network previously exported from vneTest.py
% 
% the function importONNXFunction converts an ONNX file into a function
% 
% After running this cell, a function script should appear inside the 
% same folder as this script. Four different encoding functions were 
% generated to demonstrate the flexibility of the VNE.
% 
%% 1. Trivial encoding - inputs map to outputs

encoderType = 'encoderTrivial';
modelfile = '../src/vne/models/simpleModel_ini.onnx';
paramsTrivial = importONNXFunction(modelfile, encoderType);

%% 2. Bias encoding - inputs map to outputs plus a bias

encoderType = 'encoderBias';
modelfile = '../src/vne/models/simpleModel_ini-bias.onnx';
paramsBias = importONNXFunction(modelfile, encoderType);

%% 3. 3db attenuation encoding - inputs attenuate by 3db

encoderType = 'encoder3db';
modelfile = '../src/vne/models/simpleModel_ini-3db.onnx';
params3db = importONNXFunction(modelfile, encoderType);

%% 4. Kaiming - input neurons can interact with adjacent neurons
% we do this by adding K-aiming weight initialization (see model.py) 

encoderType = 'encoderKaiming';
modelfile = '../src/vne/models/simpleModel_ini-Kaiming.onnx';
paramsKaiming = importONNXFunction(modelfile, encoderType);

%% 5. Kaiming - input neurons can interact with adjacent neurons
% we do this by adding K-aiming weight initialization (see model.py) 

encoderType = 'encoderKaimingIn';
modelfile = '../src/vne/models/simpleModel_ini-kaimingIn.onnx';
paramsKaimingIn = importONNXFunction(modelfile, encoderType);

%% 6. Kaiming - input neurons can interact with adjacent neurons
% we do this by adding K-aiming weight initialization (see model.py) 

encoderType = 'encoderKaimingOut';
modelfile = '../src/vne/models/simpleModel_ini-kaimingOut.onnx';
paramsKaimingOut = importONNXFunction(modelfile, encoderType);
%% 6. Trivial19 - attenuate a little, allow 19 inputs

encoderType = 'encoderTrivial19';
modelfile = '../src/vne/models/simpleModel_ini-Trivial19.onnx';
paramsTrivial19 = importONNXFunction(modelfile, encoderType);
%% Generate all the plots 

nsensors = 5;
nelectrodes = 5;

% in python pytorch data is stored as a 1x3 Tensor, that's what the network
% expects to receive on the input. The first dimension is a batch size, second is 
% the number sources trying to activate the output neuron, third is the number of 
% output neurons. 

Nsamples=100;
sample=zeros(1,1,nsensors);
S=zeros(Nsamples,1,nsensors);
E1=nan(Nsamples,1,nelectrodes);
E2=nan(Nsamples,1,nelectrodes);
E3=nan(Nsamples,1,nelectrodes);
E4=nan(Nsamples,1,nelectrodes);
E5=nan(Nsamples,1,nelectrodes);
E6=nan(Nsamples,1,nelectrodes);

% Encoder Validation
    close all
    figure;
    title('perturbation of the neural network to create different encoders')
    
    t=1;
    period = linspace(-1,1);
% pretend to do a dynamic measurement. record inputs and outputs
for d = linspace(1,Nsamples,Nsamples)
    %there is some linear scaling in time with some noise
    sample(1,1,:) = -5*ones(1,nsensors)+10*period(t)*ones(1,nsensors)+1.*rand(1,nsensors);
    Ts = sample(1,1,:);
    
    % Matching the formatting of the Pytorch Tensors
    S(d,1,:) = Ts;
    E1(d,1,:) = encoderTrivial(Ts,paramsTrivial);
    E2(d,1,:) = encoderBias(Ts,paramsBias);
    E3(d,1,:) = encoder3db(Ts,params3db);
    E4(d,1,:) = encoderKaiming(Ts,paramsKaiming);
    E5(d,1,:) = encoderKaimingIn(Ts,paramsKaimingIn);
    E6(d,1,:) = encoderKaimingOut(Ts,paramsKaimingOut);
    t=t+1;
end

% Plot this plot makes a big tiling of different encoder
% configurations
for s = 1:nsensors
    subplot(6,5,0*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E1(:,:,s)),'.');
    subplot(6,5,1*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E2(:,:,s)),'.');
    subplot(6,5,2*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E3(:,:,s)),'.');
    subplot(6,5,3*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E4(:,:,s)),'.');
    subplot(6,5,4*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E5(:,:,s)),'.');
    subplot(6,5,5*nsensors+s);
    scatter(squeeze(S(:,:,s)), squeeze(E6(:,:,s)),'.');
    
end
%% Use ''encoderTest'' to verify the softare is installed correctly 
%  through, no interaction). Add it to the setup and verify a small change
%  to the delay. 

% Plot S vs E1 to verify it is a straight line (S ~ E)
% figure;
% scatter(squeeze(S(:,1,3)), squeeze(E1(:,1,3)));
%%  function encoderTest.m does the trivial encoding mapping (S->E)
% use this funtion for next tests. You should be able to work with what is
% in this cell, as long as 
load("paramsAll.mat")
s=[1,22,333,444444,55555555]; % 5 sensors with 1 sample per sensor
e=encoderTest(s,paramsTrivial);
f = e-s;
%% function encoderTest19.m does a slightly non-trivial mapping (attenuates a bit) 
% like the previous encoder, but we have 19 inputs not in case there is a
% desire to test with a the DEKA LUKE prosthetic that has 19 sensors
load("paramstrivial19.mat")
% suppose we only care about 5 sensors, but loading all 19 is fine
S=zeros(1,19);
S(1)=1.2;
S(2)=2.2;
S(3)=3.3;
S(4)=4.4;
S(5)=5.5;
encoderTest19(S,paramsTrivial19); % load the routing first...
for i = 1:10
    tic
    E=encoderTest19(S,paramsTrivial19);
    d=E-S;
    toc
end






