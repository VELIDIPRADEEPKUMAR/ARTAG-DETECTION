clc;clear;clear all;
in = imread('Puffin.jpg');
imshow(in)
in1 = imread('Scene.jpg');
imshow(in1)
in2 = imread('Lena.png');
imshow(in2)
in3 = in(1:2:end,1:2:end,:); % decimation(zoom in) 800x800x3 to 400x400x3
in4 = imresize(in3,[800,800]); % interpolation , it losses some data (zoom out)

%figure ,imshow(in) ; figure , imshow(in3); 

x = imread('peppers.png');
h = [0 1 0; 1 -4 1; 0 1 0];   % edge 
h1 =[1 1 1; 1 1 1; 1 1 1];
%y = imfilter(x,h,'replicate');
figure ,imshow(x);

net = alexnet;  % neural network 
i = imread('cat.png');
imshow(i)
classify(net,imresize(i,[227,227]));


