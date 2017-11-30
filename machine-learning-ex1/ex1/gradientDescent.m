function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %approach 1 --iterative
%     for i = 1 : m
%     temp1 = theta(1) - alpha *(1/m)*(theta(1)+ theta(2)*X(i,2) - y(i))* X(i,1);
%     temp2 = theta(2) - alpha *(1/m)*(theta(1)+ theta(2)*X(i,2) - y(i))* X(i,2);
%     theta(1) = temp1;
%     theta(2) = temp2;%simultaneous updates
%     sprintf('%s','theta1');
%     disp(temp1);
%     sprintf('%s','theta2');
%     disp(temp2);
%     end   
   %approach 2-vectorized
%      temp1 = theta' * X';
%      temp2= temp1';
%      temp3 = X(:,2);
%      temp4 = alpha *(1/m)*(temp2 -y)* temp3';
%      temp = theta - temp4';%matrix dimensions problem
% 
%      theta = temp;
    % ============================================================
%     n = size(theta);%number of features
%     temp =zeros(n);
%     
%     for j = 1 : n
%      temp1 = theta' * X';
%      temp2= temp1';
%      temp5 = X(:,j)';
%      temp6 = alpha*(1/m)*(temp2 -y) * temp5;
%      temp7 = sum(temp6);
%      temp8 = sum(temp7);
%      temp(j) = theta(j)- temp8;
%      theta(j) = temp(j);
%     end

%FROM the programming tutorial
% 1 - The hypothesis is a vector, formed by multiplying the X matrix and the theta vector.
%X has size (m x n), and theta is (n x 1), so the product is (m x 1). That's good, because it's the same size as 'y'. Call this hypothesis vector 'h'.
hypothesis =  X * theta;

% 2 - The "errors vector" is the difference between the 'h' vector and the 'y' vector.
error_vector = hypothesis - y;

% 3 - The change in theta (the "gradient") is the sum of the product of X and the "errors vector", 
%scaled by alpha and 1/m. Since X is (m x n), and the error vector is (m x 1), and the result you want is the same size as theta (which is (n x 1), 
%you need to transpose X before you can multiply it by the error vector.
scalar_multiple = alpha *(1/m);
change_in_theta = scalar_multiple * X' * error_vector;%MISSED THIS!
% The vector multiplication automatically includes calculating the sum of the products.
% When you're scaling by alpha and 1/m, be sure you use enough sets of parenthesis to get the factors correct.
 
% 4 - Subtract this "change in theta" from the original value of theta. A line of code like this will do it:
theta = theta - change_in_theta;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % Print the Cost function value for every iteration
end

end
