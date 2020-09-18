clc; clear;

A = imread('20191003_121225.jpg');
figure; imshow(A); title('Original Image');
% Calculate superpixels of the image.
[L,N] = superpixels(A,100000); %best is 1000
% Display the superpixel boundaries overlaid on the original image.
BW = boundarymask(L);
%subplot(4,3,2); imshow(imoverlay(A,BW,'cyan'),'InitialMagnification',67); title ('Superpixel boundaries overlaid on the original image');
hold on; figure; imshow(imoverlay(A,BW,'cyan'),'InitialMagnification',67); title ('Superpixel boundaries overlaid on the original image');

%Set the color of each pixel in the output image to the mean RGB color of the superpixel region.
outputImage = zeros(size(A),'like',A);
idx = label2idx(L);
numRows = size(A,1);
numCols = size(A,2);
for labelVal = 1:N
    redIdx = idx{labelVal};
    greenIdx = idx{labelVal}+numRows*numCols;
    blueIdx = idx{labelVal}+2*numRows*numCols;
    outputImage(redIdx) = mean(A(redIdx));
    outputImage(greenIdx) = mean(A(greenIdx));
    outputImage(blueIdx) = mean(A(blueIdx));
end

hold on; figure; imshow(outputImage,'InitialMagnification',67)

lab_outputImage = rgb2lab(outputImage);
ab = lab_outputImage(:,:,2:3);
ab = im2single(ab);
nColors = 3;

l = lab_outputImage(:,:,1);
a = lab_outputImage(:,:,2);
b = lab_outputImage(:,:,3);

rgb_outputImage = outputImage;
r = rgb_outputImage(:,:,1);
g = rgb_outputImage(:,:,2);
b = rgb_outputImage(:,:,3);

hsv_outputImage = rgb2hsv(outputImage);
h = hsv_outputImage(:,:,1);
s = hsv_outputImage(:,:,2);
v = hsv_outputImage(:,:,3);

ycbcr_outputImage = rgb2ycbcr(outputImage);
y = ycbcr_outputImage(:,:,1);
cb = ycbcr_outputImage(:,:,2);
cr = ycbcr_outputImage(:,:,3);

B = labeloverlay (A,L);
%subplot(4,3,3); imshow(B); title('Image Overlay');

%K-MEANS
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
%subplot(4,3,4); imshow(pixel_labels,[]); title('Image Labeled by Cluster Index');

mask1 = pixel_labels==2;    %change number of mask
cluster1 = outputImage .* uint8(mask1);
hold on; figure; imshow(cluster1); title('Objects in Cluster 1');

%remove unconnected pixels
cluster1 = rgb2gray(cluster1);
BW2 = bwareaopen(cluster1, 1000); % removes objects comprised of < 150 pixels

% %fill interior gaps
% BWdfill = imfill(BW2,'holes');
% subplot(3,3,4); imshow(BWdfill); title('Binary Image with Filled Holes');

%remove connected objects on border
% BWnobord = imclearborder(BW2,26);
% subplot(4,3,6); imshow(BWnobord); title('Cleared Border Image');

%remove disconnected pixels
%Filter image, retaining only those objects with areas between 40 and 50.
%BW3 = bwareafilt(BWnobord,[40 50]);
BW3 = bwareafilt(BW2,1,'largest');
%subplot(4,3,7); imshow(BW3); title('Removed disconnected pixels');

%smooth the object
seD = strel('diamond',1);
BWfinal = imerode(BW3,seD);
BWfinal = imerode(BWfinal,seD);
%hold on; figure; imshow(BWfinal); title('Smoothen-Segmented Image');
bnwimg = BWfinal;

Ioverlay = labeloverlay(A,BWfinal);
%hold on; figure; imshow(Ioverlay); title('Mask Over Original Image')

mask = BWfinal;

% red(mask) = 0;
% green(mask) = 255;
% blue(mask) = 0;
% %recombine all channels
% newmask = cat(3, red, green, blue);
% %subplot(4,3,9); imshow(newmask);

% Mask out the background, leaving only the leaf.
% Mask the image using bsxfun() function
maskedRgbImage = bsxfun(@times, A, cast(~BWfinal, 'like', A));
% Display the mask image.
%hold on; figure; imshow(maskedRgbImage); title('Background-Only Image');

% Mask out the leaf, leaving only the background.
% Mask the image using bsxfun() function
maskedRgbImage = bsxfun(@times, A, cast(BWfinal, 'like', A));
% Display the mask image.
%hold on; figure; imshow(maskedRgbImage); title('Leaf-Only Image');

% %LEAF AREA
% %BW = imread('mask');
leafarea = bwarea(mask);

%LEAF PERIMETER
% leafperim = bwperim(mask);
% subplot(4,3,12); imshow(leafperim); title('Leaf Perimeter');

%LEAF SKELETONIZATION
%VEINS
%patabain muna
out = bwskel(BWfinal);
%subplot(4,3,12); imshow(out); title('Leaf Skeletonization');
%EDGE/BOUNDARIES
%patabain muna
leaf_boundaries = edge(BWfinal,'Sobel');

%hold on; figure; imshow(out); title('Leaf Veins');
%hold on; figure; imshow(leaf_boundaries); title('Leaf Boundaries');

%hold on; figure; imshow(imfuse(out,leaf_boundaries)); title('Lettuce Plant Skeleton');


%SEGMENTS EACH LEAF
bin_img = BWfinal;
L = watershed(bin_img);
Lrgb = label2rgb(L);
hold on; figure; imshow(Lrgb); title('First Watershed');
% use imfuse to show these two images together, zooming in on one particular blob.
%hold on; figure; imshow(imfuse(bin_img,Lrgb)); title('Fused Watershed Image'); axis([10 175 15 155])
%clean small noises
bw2 = ~bwareaopen(~bin_img, 50);  %from 10
leaf_binimg = bw2;
%hold on; figure; imshow(bw2); title('Cleaned Complement Image');
%distance transform
D = -bwdist(~bin_img);
%hold on; figure; imshow(D,[]); title('Distance Transform 1');
dist_trans = D;
%Compute the watershed transform
Ld = watershed(D);
hold on; figure; imshow(label2rgb(Ld)); title('Second Wastershed');
%Ridge lines segmentation of binary image
bw2 = bin_img;
bw2(Ld == 0) = 0;
%hold on; figure; imshow(bw2); title('Ridge Lines Segmentation');
%imextendedmin should ideally just produce small spots that are roughly in
%the middle of the cells to be segmented. imshowpair to superimpose the mask on the original image.
mask = imextendedmin(D,80);
%hold on; figure; imshowpair(bin_img,mask,'blend'); title('Cell Centroids Superimposition');
% Modify the distance transform so it only has minima at the desired locations, and then repeat the watershed steps
D2 = imimposemin(D,mask);
Ld2 = watershed(D2);
bw3 = bin_img;
bw3(Ld2 == 0) = 0;
hold on; figure; imshow(bw3); title('Third Watershed');

%%%%%%%%next phase
%Calculate the distance transform of the complement of the binary image. The value of each pixel in the output image is the distance between that pixel and the nearest nonzero pixel of bw.
D = bwdist(~bw3);
%hold on; figure; imshow(D,[]); title('Distance Transform 2 of Binary Image');
%Take the complement of the distance transformed image so that light pixels represent high elevations and dark pixels represent low elevations for the watershed transform.
D = -D;
%hold on; figure; imshow(D,[]); title('Complement of Distance Transform');
%Calculate the watershed transform. Set pixels that are outside the ROI to 0.
L = watershed(D,8);
L(~bw3) = 0;
%Display the resulting label matrix as an RGB image.
rgb = label2rgb(L,'jet',[.5 .5 .5]);
hold on; figure; imshow(rgb); title('Fourth Watershed')

%%%%%%%%iteration: 2

dist_trans = D;
%Compute the watershed transform
Ld = watershed(D);
hold on; figure; imshow(label2rgb(Ld)); title('Fifth Wastershed');
%Ridge lines segmentation of binary image
bw2 = bw3;   %consider inputting the last binary image before this section
bw2(Ld == 0) = 0;
%hold on; figure; imshow(bw2); title('Ridge Lines Segmentation');
%imextendedmin should ideally just produce small spots that are roughly in
%the middle of the cells to be segmented. imshowpair to superimpose the mask on the original image.
mask = imextendedmin(D,80);
%hold on; figure; imshowpair(bw3,mask,'blend'); title('Cell Centroids Superimposition');
% Modify the distance transform so it only has minima at the desired locations, and then repeat the watershed steps
D2 = imimposemin(D,mask);
Ld2 = watershed(D2);
bw4 = bw3;
bw4(Ld2 == 0) = 0;
hold on; figure; imshow(bw4); title('Sixth Watershed');

[labeledImage, numberOfObject] = bwlabel(bw4);
%hold on; figure; imshow(labeledImage,[]); title('Color-labeled Leaf Segmentation');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%MORPHOLOGICAL STATS FOR EACH SEGMENTED LEAF
stats_leaf = regionprops('table',bw4,'Area', 'Centroid', 'ConvexArea', 'ConvexHull', 'Extrema','FilledArea',...
    'MajorAxisLength','MinorAxisLength', 'Perimeter', 'Solidity') %'MaxFeretDiameter','MaxFeretAngle','MaxFeretCoordinates')

%MORPHOLOGICAL STATS FOR THE WHOLE BIOMASS
stats_whole = regionprops('table',bnwimg,'Area', 'Centroid', 'ConvexArea', 'ConvexHull', 'Extrema','FilledArea',...
    'MajorAxisLength','MinorAxisLength', 'Perimeter', 'Solidity') %'MaxFeretDiameter','MaxFeretAngle','MaxFeretCoordinates')

%MORPHOLOGICAL STATS FOR CONVEXHULL OF THE BIOMASS
ch_whole = bwconvhull(bnwimg);
%hold on; figure; imshow(ch_whole); title('ConvexHull Image');
stats_convexhull = regionprops('table',ch_whole,'Area', 'Perimeter', 'MajorAxisLength')

%leafperim = stats_whole.Area;
fill_binimg = imfill(bnwimg, 'holes');
perim_convexhull = stats_convexhull.Perimeter;
%biomass_convexity = perim_biomass / perim_convexhull



% I = fitsread('solarspectra.fts');
% imshow(maskedRgbImage,[]);
% improfile
% 
% It = bwmorph(out,'thin','inf');
% B = bwmorph(It,'branchpoints');
% [i,j] = find(bwmorph(It,'endpoints'));
% D = bwdistgeodesic(It,find(B),'quasi');
% imshow(out);
% for n = 1:numel(i) text(j(n),i(n),[num2str(D(i(n),j(n)))],'color','g');
% end



% DT = delaunayTriangulation(x(:),y(:));
% [H,A] = convexHull(DT);

% idx = find([stats.Area] > 80);
% BW2 = ismember(labelmatrix(bw4), idx);
%
% stats = regionprops(L, 'Area');
% allArea = [stats.Area];

% %Calculate the distance transform of the complement of the binary image. The value of each pixel in the output image is the distance between that pixel and the nearest nonzero pixel of bw.
% D4 = bwdist(~bw4);
% hold on; figure; imshow(D4,[]); title('Distance Transform 2 of Binary Image');
% %Take the complement of the distance transformed image so that light pixels represent high elevations and dark pixels represent low elevations for the watershed transform.
% D4 = -D4;
% hold on; figure; imshow(D4,[]); title('Complement of Distance Transform');
% %Calculate the watershed transform. Set pixels that are outside the ROI to 0.
% L4 = watershed(D4,8);
% L4(~bw4) = 0;
% %Display the resulting label matrix as an RGB image.
% rgb4 = label2rgb(L4,'jet',[.5 .5 .5]);
% hold on; figure; imshow(rgb4); title('Seventh Watershed')
%
% %%%%%%%%iteration: 3
%
% dist_trans = D4;
% %Compute the watershed transform
% Ld = watershed(dist_trans);
% hold on; figure; imshow(label2rgb(Ld)); title('Eigth Wastershed');
% %Ridge lines segmentation of binary image
% bw2 = bw4;   %consider inputting the last binary image before this section
% bw2(Ld == 0) = 0;
% hold on; figure; imshow(bw2); title('Ridge Lines Segmentation');
% %imextendedmin should ideally just produce small spots that are roughly in
% %the middle of the cells to be segmented. imshowpair to superimpose the mask on the original image.
% mask = imextendedmin(dist_trans,80);
% hold on; figure; imshowpair(bw4,mask,'blend'); title('Cell Centroids Superimposition');
% % Modify the distance transform so it only has minima at the desired locations, and then repeat the watershed steps
% D2 = imimposemin(dist_trans,mask);
% Ld2 = watershed(D2);
% bw5 = bw4;
% bw5(Ld2 == 0) = 0;
% hold on; figure; imshow(bw5); title('Ninth Watershed');
%


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %SOM NEURAL NETWORK
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a_lin = a(:);
% b_lin = b(:);
% ab_lin = horzcat(a_lin, b_lin);
% ab_lin_uniq = unique(ab_lin);
% ab_lin_uniq_trans = ab_lin_uniq';
% plot(ab_lin_uniq_trans(1,:),ab_lin_uniq_trans(2,:),'+r'); hold on;
% X = ab_lin_uniq_trans;
%
% net = selforgmap([5 6]); %number of neurons for each layer (there are two layers)
% net = configure(net,X);
%
% plotsompos(net); hold on;
%
% % train for number of epochs
% net.trainParam.epochs = 1;
% net = train(net,X);
% plotsompos(net)
%
% %set test data to x
% x = [0.5;0.3];
% y = net(x)