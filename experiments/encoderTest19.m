function encoded_data = encoderTest19(sensor_data, paramsTrivial19)
% function reads in a (1,19) array of sensor data and encodes the samples using the 
% Trivial encoder. The result is an encoded_data (1,19) array that will be very close to the 
% original sensor_data.

% sensor_data=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
Ts = zeros(1,1,19); %sampleTensor is like a container for the one sample. It is formatted like this becuse of the way Pytorch sets up the neural network. 


Ts(1,1,:)=sensor_data;%load the data samples into the container

Te = encoderTrivial19(Ts,paramsTrivial19);% run it through the network

encoded_data = transpose(squeeze(Te(1,1,:)));%unload the result

end