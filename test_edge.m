I1 = rgb2gray(imread('large.png'));
I = im2bw(I1,0.5);

BW = edge(I,'Sobel',0.01);    % edge detection 
F = imfill(BW,'holes');   % fill the image 

%Fill a gap 
se = strel('disk',2);
F = imclose(F,se);


figure ;
subplot(2,2,1);
imshow(I);
subplot(2,2,2);
imshow(BW);
subplot(2,2,3);
F = bwareaopen(F, 5000);
imshow(F);

B = regionprops(F,'BoundingBox');
Xl = int16(B.BoundingBox(1));
Yl = int16(B.BoundingBox(2));
ah = int16(B.BoundingBox(3));
av = int16(B.BoundingBox(4));

P = regionprops(F,'Perimeter');

C = regionprops(F,'Centroid');

XY = C.Centroid;

centroids = cat(1,C.Centroid);
subplot(2,2,4);
imshow(I)
hold on
plot(Xl,Yl,'b*')
plot(centroids(1)+(0:20:int16(ah/2)),centroids(2),'b*')
viscircles(centroids,ah/2);
hold off


D = (zeros(7,7));

figure ;
%subplot(1,2,1);
imshow(I);
hold on
plot(Xl,Yl,'b*')
hold off


x = Xl;

for i = 1:7
    
   y = Yl;
    for j = 1:7 
        D(j,i) = 1;
        p1 = int16(ah/14);
        p2 = int16(ah/14);
        D(j,i) = I(y+p2,x+p1 );
        
        if D(j,i)
        hold on
        plot(x+int16(ah/14) , y+int16(ah/14),'b*');
        hold off
        else 
        hold on
        plot(x+int16(ah/14) , y+int16(ah/14),'r*');
        hold off
        end
            
         y = y + ah/7;
    end
    
    x = x + ah/7;

end