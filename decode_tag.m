function [D] = decode_tag(I1,N)

T_Gaus= imgaussfilt(I1);
I = im2bw(T_Gaus,0.2);

BW = edge(I,'Sobel');    % edge detection 
F = imfill(BW,'holes');   % fill the image 

se = strel('disk',2);
F = imclose(F,se);

% figure ;
% subplot(2,2,1);
% imshow(I);
% subplot(2,2,2);
% imshow(BW);
% subplot(2,2,3);
% F = bwareaopen(F, 5000);
% imshow(F);

B = regionprops(F,'BoundingBox');
L = length(B);

Xl = zeros(1,L);
Yl = zeros(1,L);
ah = zeros(1,L);
av = zeros(1,L);

for i = 1:L
    
Xl(i) = int16(B(i).BoundingBox(1));
Yl(i) = int16(B(i).BoundingBox(2));
ah(i) = int16(B(i).BoundingBox(3));
av(i) = int16(B(i).BoundingBox(4));

% figure ;
% CROP = imcrop (I, [B(i).BoundingBox]);
% imshow(CROP);

end

D = (zeros(N,N,L));

figure ;
%subplot(1,2,1);
imshow(I);
hold on
plot(Xl,Yl,'b*')
hold off

for k = 1:L

x = Xl(k);

for i = 1:N
    
   y = Yl(k);
    for j = 1:N 
        D(j,i,k) = 1;
        p1 = int16(ah(k)/(2*N));
        p2 = int16(av(k)/(2*N));
        D(j,i,k) = I(y+p2,x+p1 );
        
        if D(j,i,k)
        hold on
        plot(x+int16(ah(k)/(2*N)) , y+int16(av(k)/(2*N)),'b*');
        hold off
        else 
        hold on
        plot(x+int16(ah(k)/(2*N)) , y+int16(av(k)/(2*N)),'r*');
        hold off
        end
            
         y = y + int16(av(k)/N);
    end
    
    x = x + int16(ah(k)/N);

end

end

end
