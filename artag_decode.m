I1 = rgb2gray(imread('1.png'));

N = 7; % N is the no of squares in artag in each column or row 

D = decode_tag(I1,N);  

