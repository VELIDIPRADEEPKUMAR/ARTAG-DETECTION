RGB = imread('braille.jpg');
I = rgb2gray(RGB);
I2 = imcrop(I,[14.5100 3.5100 189.9800 134.9800]);
h = ones(5,5) / 25;
I3=imfilter(I2,h);
I4 = imadjust(I3);
I5 = imcomplement(I4);
se = strel('disk',1);
I6 = imdilate(I5,se);
I7 = im2bw(I6, 0.4);
subplot(4,4,1)
imshow(RGB)
title('rgb')
subplot(4,4,2)
imshow(I)
title('gray Image')
subplot(4,4,3)
imshow(I2)
title('Cropped Image')
subplot(4,4,4)
imshow(I3)
title('filtered Image')
subplot(4,4,5)
imshow(I4)
title('contrast Image')
subplot(4,4,6)
imshow(I5)
title('complement Image')
subplot(4,4,7)
imshow(I6)
title('dilate Image')