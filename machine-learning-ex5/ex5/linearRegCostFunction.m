function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n_plus_1 = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

temp1 = theta'* X';
temp2 = temp1';
%theta without first term
theta_without_first_term = theta(2 : n_plus_1);
% theta square
square_theta = theta_without_first_term.^2;
%sum
sum_square_theta = sum(square_theta);
%[1 x 1] = (n+1) x 1
reg_term = (lambda / ( 2 * m)) * sum_square_theta;

temp3 = 0.5*(1/m)*(temp2 - y ).^2;
J = sum(temp3) + reg_term;


% error vector [m x 1]
error_vector = temp2 - y;
%gradient computation

%first term
%[1 x 1] = (1 x m) x (m x 1)
grad(1) = (1/m)* X(:,1)' * error_vector;

%other terms
% [(n) x 1] = [(n) x m]x [m x 1]
grad_term1 = (1/m)* X(:,2:n_plus_1)' * error_vector;

grad_term2 = lambda*(1/m)*theta(2: n_plus_1);
grad(2: n_plus_1) = grad_term1 + grad_term2;


% =========================================================================

grad = grad(:);

end
