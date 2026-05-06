function pano = stitch_two_images()
% STITCH_PANORAMA_SCALED
% Create a panorama using downscaled images (compatible with older MATLAB versions).
% Reduces memory usage by scaling input images.
 
%% User Parameters
scaleFactor = 0.3;  % <-- Decrease if you have very low RAM (e.g., 0.25)
files = { ...
    'IMG_7984.jpg', ...
    'IMG_7985.jpg', ...
    'IMG_7986.jpg' ...
    };
 
%% 1. Read and Scale Images
imgs = cell(size(files));
for i = 1:numel(files)
    I = im2double(imread(files{i}));
    imgs{i} = imresize(I, scaleFactor);
end
 
%% 2. Convert to Grayscale
grays = cellfun(@(I) rgb2gray(im2uint8(I)), imgs, 'UniformOutput', false);
 
%% 3. Detect SURF Features
points = cellfun(@(G) detectSURFFeatures(G, 'MetricThreshold', 1000), grays, 'UniformOutput', false);
[feats, vpts] = cellfun(@(G,P) extractFeatures(G, P.selectStrongest(2000)), grays, points, 'UniformOutput', false);
 
%% 4. Initialize Transforms
num = numel(imgs);
centerIdx = 2; % use middle image as reference
tforms(num) = projective2d(eye(3)); % initialize structure array
tforms(centerIdx) = projective2d(eye(3));
 
%% 5. Compute Transformations (left and right)
for k = [centerIdx-1, centerIdx+1]
    if k < 1 || k > num, continue; end
 
    if k < centerIdx
        a = k; b = k+1; % map left -> center
    else
        a = k; b = k-1; % map right -> center
    end
 
    idxPairs = matchFeatures(feats{a}, feats{b}, 'Unique', true, ...
        'MaxRatio', 0.7, 'MatchThreshold', 80);
    matchedA = vpts{a}(idxPairs(:,1));
    matchedB = vpts{b}(idxPairs(:,2));
 
    [t_ab, inliers] = estimateGeometricTransform2D(matchedA, matchedB, ...
        'projective', 'MaxNumTrials', 3000, 'Confidence', 99.9, 'MaxDistance', 3);
    fprintf('Image %d to %d: %d inliers\n', a, b, nnz(inliers));
 
    % chain transforms
    tforms(a) = projective2d(t_ab.T * tforms(b).T);
end
 
%% 6. Define Output Limits
imageSize = size(grays{1});
[xLimits, yLimits] = deal(zeros(num,2), zeros(num,2));
for i = 1:num
    [xLimits(i,:), yLimits(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end
 
xMin = floor(min(xLimits(:))); xMax = ceil(max(xLimits(:)));
yMin = floor(min(yLimits(:))); yMax = ceil(max(yLimits(:)));
width  = xMax - xMin + 1;
height = yMax - yMin + 1;
R = imref2d([height width], [xMin xMax], [yMin yMax]);
 
%% 7. Initialize Panorama and Blend
pano = zeros(height, width, 3);
weight = zeros(height, width);
 
for i = 1:num
    warped = imwarp(imgs{i}, tforms(i), 'OutputView', R);
    mask = imwarp(ones(size(grays{i}), 'like', grays{i}), tforms(i), 'OutputView', R) > 0;
    d = bwdist(~mask);
    w = mat2gray(d);
 
    for c = 1:3
        plane = warped(:,:,c);
        plane(~mask) = 0;
        pano(:,:,c) = pano(:,:,c) + plane .* w;
    end
    weight = weight + w;
end
 
weight(weight == 0) = 1;
for c = 1:3
    pano(:,:,c) = pano(:,:,c) ./ weight;
end
 
imshow(pano), title('Panorama (Scaled)');
imwrite(pano, 'panorama_scaled_output.jpg');
fprintf('Panorama saved to panorama_scaled_output.jpg\n');
end