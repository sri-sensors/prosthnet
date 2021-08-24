function encoded_data = encoderTest(sensor_data, paramsTrivial)
% function reads in a (1,5) array of sensor data and encodes the samples using the 
% Trivial encoder. The result is an encoded_data (1,5) array that will be very close to the 
% original sensor_data.

% sensor_data=[1,2,3,4,5];
Ts = zeros(1,1,5); %sampleTensor is like a container for the one sample. It is formatted like this becuse of the way Pytorch sets up the neural network. 


Ts(1,1,:)=sensor_data;%load the data samples into the container

Te = encoderTrivial(Ts,paramsTrivial);% run it through the network

encoded_data = Te(1,1,:);%unload the result

end