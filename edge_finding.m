


%h = fspecial('Canny', 50, 45);
%imfilter(A,h,'conv')
%BW2 = edge(I,'Prewitt');
%I = imread('10.png');
I = rgb2gray(imread('10.png'));
I = im2bw(I,0.5);

BW1 = edge(I,'Canny');
BW2 = edge(I,'Prewitt');
BW3 = edge(I,'Roberts');
BW4 = edge(I,'Sobel');

figure;
imshow(I);

% edge detection 
figure ;
subplot(2,2,1);
imshow(BW1);
title('Canny');
subplot(2,2,2);
imshow(BW2);
title('prewitt');
subplot(2,2,3);
imshow(BW3);
title('roberts');
subplot(2,2,4);
imshow(BW4);
title('sobel');

%fill the holes ;

F1 = imfill(BW1,'holes');
F2 = imfill(BW2,'holes');
F3 = imfill(BW3,'holes');
F4 = imfill(BW4,'holes');

figure ;
subplot(2,2,1);
imshow(F1);
title('Canny');
subplot(2,2,2);
imshow(F2);
title('prewitt');
subplot(2,2,3);
imshow(F3);
title('roberts');
subplot(2,2,4);
imshow(F4);
title('sobel');

% image region analyzer  





