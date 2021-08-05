clear;
clc;
Ir = imread('large.png');
%Ir = imread('oh3hi.jpg');
%Ir = imread('tag_cubes.jpg');
I1 = rgb2gray(Ir);

T_Gaus= imgaussfilt(I1);
I = imbinarize(T_Gaus,0.3);

BW = edge(I,'Sobel');    % edge detection 
se1 = strel('disk', 5);
se2 = strel('disk', 4);
BW = imdilate(BW,se1);
BW = imerode(BW,se2);
F = imfill(BW,'holes');   % fill the image 

%Camera calibration matrix
fx = 1406.08415449821;
fy = 1417.99930662800;
cx = 1014.13643417416;
cy = 566.347754321696;
s = 2.20679787308599;
% fx = 629.302552;
% fy = 635.529018;
% cx = 960;
% cy = 1000;
% s = 0;
alf = 1;

% K =[1406.08415449821,0,0;2.20679787308599, 1417.99930662800,0;1014.13643417416,566.347754321696,1]';
K = [fx s cx;0 fy cy;0 0 1];

%Fill a gap 
se = strel('disk',2);
F = imclose(F,se);

%F1 = regionprops(F,'FilledImage');
% plotting the basic image filters  'FilledImage'

%----------------------------------------------------------------------------
%Concentrate only on the exterior boundaries. Option 'noholes' will accelerate the processing by preventing bwboundaries from searching for inner contours.
%[B,L] = bwboundaries(bw,'noholes');

%Display the label matrix and draw each boundary.
%imshow(label2rgb(L,@jet,[.5 .5 .5]))
%hold on
%for k = 1:length(B)
 % boundary = B{k};
  %plot(boundary(:,2),boundary(:,1),'w','LineWidth',2)
%end

%********************
%https://www.mathworks.com/help/images/identifying-round-objects.html
%********************
%----------------------------------------------------------------------------


figure ;
subplot(2,2,1);
imshow(I);
subplot(2,2,2);
imshow(BW);
subplot(2,2,3);
F = bwareaopen(F, 5000);
imshow(F);

% finding the tage image 

%------------------------------------------------------------------------
%Remove objects containing fewer than 50 pixels using bwareaopen function.
%BW2 = bwareaopen(BW, 50);

%Find Regions Without Holes
%BW2 = bwpropfilt(BW,'EulerNumber',[1 1]);

%Find Which Ten Objects Have Largest Perimeters
%BW2 = bwpropfilt(BW,'perimeter',10);

%Calculate properties of regions in the image and return the data in a table.
%stats = regionprops('table',bw,'Centroid','MajorAxisLength','MinorAxisLength')

%Get centers and radii of the circles.
%centers = stats.Centroid;
%diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
%radii = diameters/2;
%------------------------------------------------------------------------

%% boundries 
B1 = bwboundaries(F, 8, 'noholes'); % Boundary detection
B_size = size(B1);

figure ;
%subplot(1,2,1);
imshow(Ir);
hold on
for k = 1:B_size(1,1)
plot(B1{k}(:,2),B1{k}(:,1),'b*')
end
hold off

figure(5) ;
imshow(Ir);
%%

for k = 1:B_size(1,1)
    
    BB = B1{k};
 ps = dpsimplify(BB,10); %Douglas-Peucker Algorithm
        
        ps_size = size(ps);
        final_corners = zeros(4,2);
             if(  ps_size(1) == 5)                                              %If it is a polygon with 4 corners, then detect as quad
            for k1=1:ps_size(1)-1
                for kk=1:2 
                    final_corners(k1,kk) = ps(k1,kk); %Only four corners
                end
            end
             
        % Area filter for small noise
            maxi=max(final_corners,[],1);
            mini=min(final_corners,[],1);
            
%plot(final_corners(:,2),final_corners(:,1),'ro');
          


%% HOMOGRAPHY
             %Reference marker points
             %quad_size = size(ref_marker);
             quad_pts(1,:) = [1, 1];
             quad_pts(2,:) = [600, 1];
             quad_pts(3,:) = [600, 600];
             quad_pts(4,:) = [1, 600];
             
             %Corner points
             final_pts = [final_corners(:,2), final_corners(:,1)];
             
             
             %Estimate homography with the 2 sets of four points
             H = fitgeotrans(quad_pts, final_pts,'projective');
             invH = inv(H.T');
             H_1 = projective2d(invH');
             
             %Warp the marker to image plane
             RA = imref2d([quad_pts(3,1) quad_pts(3,2)], [1 quad_pts(3,1)-1], [1 quad_pts(3,1)-1]);
             [warp,r] = imwarp(I1, H_1, 'OutputView', RA); 
             
              figure, imshow(warp);
              
              th = graythresh(warp);
             markBin = imbinarize(warp, th);
              se3 = strel('square', 1);
              markBin = imerode(markBin,se3);
              figure, imshow(markBin);
              
              
              %% GRID CREATION AND BITS DETECTION

            %Create the 8x8 grid
            maxi=max(quad_pts,[],1);
            mini=min(quad_pts,[],1);
            p=norm(maxi(1)-mini(1));
            q=norm(maxi(2)-mini(2));
            for iii=1:8
              m(iii)=round(mini(1)+(iii-1)/8*p);
              n(iii)=round(mini(2)+(iii-1)/8*q);
            end
            Agrid=markBin;
           for iii=1:8
               for jjj=mini(2):maxi(2)
                   Agrid((m(iii)),jjj) = 1; 
               end
           end  
           for jjj=1:8
               for iii=mini(1):maxi(1)
                   Agrid(iii,n(jjj)) = 1; 
               end
           end
           figure 
           imshow(Agrid);hold on;
           m(9)=maxi(1);
           n(9)=maxi(2);
           
           %Get the bits of each of the cells of the grid
           intensity=0;
           for iii=1:8
            for jjj=1:8  
                   length=0;
                   for kk=(m(iii):m(iii+1))
                      breadth=0;
                      length=length+1;
                      for l=(n(jjj):n(jjj+1))
                      breadth=breadth+1;
                        intensity=intensity+markBin(kk,l);
                      end
                   end
                   temp=intensity;
                   intensity=intensity-temp;
                   if((temp+30)>(length*breadth))%a tolerance of ~ pixels is provided
                     check(iii,jjj)=1;
                   else
                     check(iii,jjj)=0;
                   end
            end
           end
           
 % Orientation of the marker
           % First condition: marker has two outer rows and columns of 0
           % values 
           if check(1:2,:) == zeros(2,8) 
              if check(:,1:2) == zeros(8,2)
                  if check(7:8,:) == zeros(2,8)
                      if check(:,7:8) == zeros(8,2)
                          if (check(6,6) || check(6,3) || check(3,6) || check(3,3))
                            flag = 1;
                          end
                      end
                  end
              end
           else flag = 0;
           end
           if flag == 1 
               %Look for a 1 in one of the corners of interior 4x4
               pose1 = 0; pose2 = 0; pose3 = 0; pose4 = 0;
               if (check(6,6) == 1)
                   pose1 = 1; pose2 = 0; pose3 = 0; pose4 = 0;
                   lena_pts = [final_pts(1,1),final_pts(1,2);final_pts(2,1),final_pts(2,2);final_pts(3,1),final_pts(3,2);final_pts(4,1),final_pts(4,2)];
               end
               if (check(6,3) == 1)
                   pose1 = 0; pose2 = 1; pose3 = 0; pose4 = 0;
                   lena_pts = [final_pts(2,1),final_pts(2,2);final_pts(3,1),final_pts(3,2);final_pts(4,1),final_pts(4,2);final_pts(1,1),final_pts(1,2)];
               end
               if (check(3,3) == 1)
                   pose1 = 0; pose2 = 0; pose3 = 1; pose4 = 0;
                   lena_pts = [final_pts(3,1),final_pts(3,2);final_pts(4,1),final_pts(4,2);final_pts(1,1),final_pts(1,2);final_pts(2,1),final_pts(2,2)];
               end
               if (check(3,6) == 1)
                   pose1 = 0; pose2 = 0; pose3 = 0; pose4 = 1;
                   lena_pts = [final_pts(4,1),final_pts(4,2);final_pts(1,1),final_pts(1,2);final_pts(2,1),final_pts(2,2);final_pts(3,1),final_pts(3,2)];
               end
% ID of the marker. Look for the binary coding in the 2x2 
               %First plot the corners in the correct orientation
               % Then, get the binary values for it and translate to
               % decimal to get the ID
               b = [0 0 0 0];

               if (pose1 == 1)
                   figure, plot(final_pts(1,1),final_pts(1,2),'r.','markersize',20)
                   figure, plot(final_pts(2,1),final_pts(2,2),'g.','markersize',20)
                   figure, plot(final_pts(3,1),final_pts(3,2),'b.','markersize',20)
                   figure, plot(final_pts(4,1),final_pts(4,2),'y.','markersize',20)
                   
                   b(1)=check(4,4);
                   b(2)=check(4,5);
                   b(3)=check(5,5);
                   b(4)=check(5,4);
                   id=binaryVectorToDecimal(b,'LSBFirst');
                   text_str = ['ID: ' num2str(id)];
                   t = text(final_pts(1,1)+150,final_pts(1,2),text_str,'Color','b','FontSize',12);
                   
               elseif (pose2 == 1)
                   figure, plot(final_pts(2,1),final_pts(2,2),'r.','markersize',20)
                   figure, plot(final_pts(3,1),final_pts(3,2),'g.','markersize',20)
                   figure, plot(final_pts(4,1),final_pts(4,2),'b.','markersize',20)
                   figure, plot(final_pts(1,1),final_pts(1,2),'y.','markersize',20)
                   
                   b(1)=check(4,5);
                   b(2)=check(5,5);
                   b(3)=check(5,4);
                   b(4)=check(4,4);
                   id=binaryVectorToDecimal(b,'LSBFirst');
                   text_str = ['ID: ' num2str(id)];
                   t = text(final_pts(1,1)+150,final_pts(1,2),text_str, 'Color','b','FontSize',12);
             
               elseif (pose3 == 1)
                   figure, plot(final_pts(3,1),final_pts(3,2),'r.','markersize',20)
                   figure, plot(final_pts(4,1),final_pts(4,2),'g.','markersize',20)
                   figure, plot(final_pts(1,1),final_pts(1,2),'b.','markersize',20)
                   figure, plot(final_pts(2,1),final_pts(2,2),'y.','markersize',20)
                   
                   b(1)=check(5,5);
                   b(2)=check(5,4);
                   b(3)=check(4,4);
                   b(4)=check(4,5);
                   id=binaryVectorToDecimal(b,'LSBFirst');
                   text_str = ['ID: ' num2str(id)];
                   t = text(final_pts(1,1)+150,final_pts(1,2),text_str,'Color','b','FontSize',12);

               elseif (pose4 == 1)
                   figure, plot(final_pts(4,1),final_pts(4,2),'r.','markersize',20)
                   figure, plot(final_pts(1,1),final_pts(1,2),'g.','markersize',20)
                   figure, plot(final_pts(2,1),final_pts(2,2),'b.','markersize',20)
                   figure, plot(final_pts(3,1),final_pts(3,2),'y.','markersize',20)
                   
                   b(1)=check(5,4); 
                   b(2)=check(4,4);
                   b(3)=check(4,5);
                   b(4)=check(5,5);
                   id=binaryVectorToDecimal(b,'LSBFirst');
                   text_str = ['ID: ' num2str(id)];
                   t = text(final_pts(1,1)+150,final_pts(1,2),text_str,'Color','b','FontSize',12);
               end
           end
           
           %% PROJECTION MATRIX AND CUBE
            if (pose1 == 1 || pose2 == 1 || pose3 == 1 || pose4 == 1)
            % Estimate homography with Unit Square and corner points (lena points)
            m_size=1;
            quad_pts1(:,1) = [0; 0; 1];
            quad_pts1(:,2) = [m_size; 0; 1];
            quad_pts1(:,3) = [m_size; m_size; 1];
            quad_pts1(:,4) = [0; m_size; 1];
            lena_pts1 = lena_pts';
            row_size = size(lena_pts1, 2);
            ones_row = ones(1, row_size);
            lena_pts2 = [lena_pts1; ones_row];
            Hom2 = homography2d(quad_pts1, lena_pts2);
            H2 = Hom2/Hom2(3,3);
            
            %Build the Projection Matrix with the camera intrinsic
            %parameters matrix and the homography matrix
            RT = inv(K)*H2;
            Rt(:,1) = RT(:,1);
            Rt(:,2) = RT(:,2);
            Rt(:,3) = cross(Rt(:,1),Rt(:,2));
            Rt(:,4) = RT(:,3);
            P = K * Rt;

            %Build cube
            %Top of the cube points
            x_c1 = P * [0;0;-1;1];
            x_c1 = x_c1/x_c1(3);
            line_x1=[x_c1(1) lena_pts2(1,1) ];
            line_y1=[x_c1(2) lena_pts2(2,1) ];
            figure(1), line(line_x1,line_y1,'Color','r','LineWidth',1)

            x_c2 = P * [m_size;0;-1;1];
            x_c2 = x_c2/x_c2(3);
            line_x2=[x_c2(1) lena_pts2(1,2) ];
            line_y2=[x_c2(2) lena_pts2(2,2) ];
            figure(1), line(line_x2,line_y2,'Color','r','LineWidth',1)

            x_c3 = P * [m_size;m_size;-1;1];
            x_c3 = x_c3/x_c3(3);
            line_x3=[x_c3(1) lena_pts2(1,3) ];
            line_y3=[x_c3(2) lena_pts2(2,3) ];
            figure(1), line(line_x3,line_y3,'Color','r','LineWidth',1)

            x_c4 = P * [0;m_size;-1;1];
            x_c4 = x_c4/x_c4(3);
            line_x4=[x_c4(1) lena_pts2(1,4) ];
            line_y4=[x_c4(2) lena_pts2(2,4) ];
            figure(1), plot(line_x4,line_y4,'Color','r','LineWidth',1)
            
            line_x5=[x_c1(1) x_c4(1) ];
            line_y5=[x_c1(2) x_c4(2) ];
            figure(1), plot(line_x5,line_y5,'Color','r','LineWidth',1)
            
            line_x6=[x_c1(1) x_c2(1) ];
            line_y6=[x_c1(2) x_c2(2) ];
            figure(1), plot(line_x6,line_y6,'Color','r','LineWidth',1)
            
            line_x7=[x_c2(1) x_c3(1) ];
            line_y7=[x_c2(2) x_c3(2) ];
            figure(1), plot(line_x7,line_y7,'Color','r','LineWidth',1)
            
            line_x8=[x_c3(1) x_c4(1) ];
            line_y8=[x_c3(2) x_c4(2) ];
            figure(1), plot(line_x8,line_y8,'Color','r','LineWidth',1)
            figure, plot(x_c1(1),x_c1(2),'m.','markersize',20)
            figure, plot(x_c2(1),x_c2(2),'c.','markersize',20)
            figure, plot(x_c3(1),x_c3(2),'g.','markersize',20)
            figure, plot(x_c4(1),x_c4(2),'r.','markersize',20)
            end
          
            
           
           
           
           
           
             end
        
             
             
             
             
end

%%
% B = regionprops(F,'BoundingBox');
% L = length(B);
% 
% Xl = zeros(1,L);
% Yl = zeros(1,L);
% ah = zeros(1,L);
% av = zeros(1,L);
% 
% for i = 1:L
%     
% Xl(i) = int16(B(i).BoundingBox(1));
% Yl(i) = int16(B(i).BoundingBox(2));
% ah(i) = int16(B(i).BoundingBox(3));
% av(i) = int16(B(i).BoundingBox(4));
% 
% % figure ;
% % CROP = imcrop (I, [B(i).BoundingBox]);
% % imshow(CROP);
% 
% end
% 
% % P = regionprops(F,'Perimeter');
% % p = [P(1).Perimeter P(2).Perimeter] ;
% % C = regionprops(F,'Centroid');
% % 
% % XY = C.Centroid;
% 
% % centroids = cat(1,C.Centroid);
% % subplot(2,2,4);
% % imshow(I)
% % hold on
% % plot(Xl,Yl,'b*')
% % plot(centroids(1)+(0:20:int16(ah/2)),centroids(2),'b*')
% % viscircles(centroids,p/8);
% % hold off
% 
% D = (zeros(7,7,L));
% 
% figure ;
% %subplot(1,2,1);
% imshow(I);
% hold on
% plot(Xl,Yl,'b*')
% hold off
% 
% 
% 
% for k = 1:L 
% x = Xl(k);
% for i = 1:7
%     
%    y = Yl(k);
%     for j = 1:7 
%         D(j,i,k) = 1;
%         p1 = int16(ah(k)/14);
%         p2 = int16(av(k)/14);
%         D(j,i,k) = I(y+p2,x+p1);
%         
%         if D(j,i,k)
%         hold on
%         plot(x+int16(ah(k)/14) , y+int16(av(k)/14),'b*');
%         hold off
%         else 
%         hold on
%         plot(x+int16(ah(k)/14) , y+int16(av(k)/14),'r*');
%         hold off
%         end
%             
%          y = y + int16(av(k)/7);
%     end
%     
%     x = x + int16(ah(k)/7);
% 
% end
% 
% end








