clear;
clc;

cam = webcam ;
preview(cam);
cam.AvailableResolutions ;
cam.Resolution = '320x240';
%closePreview(cam);
k = 0;
figure;
img = snapshot(cam);
IR = rgb2gray(img);

while 1
    
img = snapshot(cam);
%clear('cam');
k = k + 1;
%I1 = rgb2gray(img);
if(k>9)
IR = img;
k = 0;
end
I2 = IR - img;
imshow(I2);
%imshow(mat2gray(double(I1).*double(imbinarize(I2,0.5))));

end