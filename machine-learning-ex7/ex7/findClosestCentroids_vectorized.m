function idx = findClosestCentroids_vectorized(X, centroids)
%FINDCLOSESTCENTROIDS_VECTORIZED vectorized implementation
%#of training examples
m =  size(X,1);
%# of features
n = size(X,2)

% Set K, K is the number of centroids.
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

%vectorized implementation
% Create a "distance" matrix of size (m x K) and initialize it to all zeros.
distance_matrix = zeros(m, K);
% Use a for-loop over the 1:K centroids.
for k = 1:K
% Inside this loop, create a column vector of the distance from each training example 
% to that centroid, and store it as a column of the distance matrix. 
% One method is to use the bsxfun) function and the sum() function 
% to calculate the sum of the squares of the differences between each row in the X matrix and a centroid.
% When the for-loop ends, you'll have a matrix of centroid distances.
% Then return idx as the vector of the indexes of the locations with the minimum distance.
% The result is a vector of size (m x 1) with the indexes of the closest centroids.
diff = bsxfun(@minus, X, centroids(k,:));
distance_matrix(:,k) = sum(diff.^2,2);

end
 %find minimum
[M,I] = min(distance_matrix,[],2);
idx = I;

end