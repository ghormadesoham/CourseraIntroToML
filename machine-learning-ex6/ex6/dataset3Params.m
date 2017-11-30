function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30;];
% length of the vector
length_test_val = length(test_values);
% result 
result_matrix =  zeros(length_test_val^2 ,3);% 3 params -C,sigma and error

for i = 1: length_test_val
    C_temp = test_values(i);
    
    for j = 1: length_test_val
        
    sigma_temp = test_values(j);
    %train on training set
    model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
    %use svmPredict to predict the labels on the cross validation set.
     predictions = svmPredict(model, Xval);
     pred_error = mean(double(predictions ~= yval));
     %populate the results matrix
     result_matrix(length_test_val * (i - 1) + j,:) = [C_temp sigma_temp pred_error];
     
    end
end

% find the minimum pred_error and set the corresponding values of C and
% sigma 
[M,I] = min(result_matrix);
min_index = I(1,3);
C = result_matrix(min_index,1);
sigma = result_matrix(min_index,2);

% =========================================================================

end
