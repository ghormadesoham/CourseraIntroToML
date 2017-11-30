function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
%#of training examples
m =  size(X,1);
%# of features
n = size(X,2)

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% loop over training examples
for i = 1 : m
    %result matrix with  columns for index j and norm value
    results = zeros(K,2);
    % loop over centroids
for j  = 1 : K
    diff = (X(i,:) - centroids(j,:));
    sq_diff = diff.^2;  
    norm = sum(sq_diff);
    results(j,:) = [ j norm  ];
end
%find minimum
 [M,I] = min(results);
 % norm is in the second column ;find the index for the minimum value of
 % norm
 index_of_min_norm = I(1,2);
 %find the corresponding index for the centroid with the atleast norm
centroid_index = results(index_of_min_norm,1);
 idx(i) = centroid_index;
end





% =============================================================

end

