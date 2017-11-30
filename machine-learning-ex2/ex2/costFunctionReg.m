function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
size_theta = size(theta);
n_plus_1 = size_theta;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size_theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%[ m  x 1 ] = [m x (n + 1)] x [(N+1) x 1]
z = X * theta;
%[ m x 1 ]
logOfSigmoid =  log(sigmoid(z));
%[ 1 x 1 ] = [ 1 x m ] x [m x 1]
term1 = logOfSigmoid' * y;
%[ m x 1]
logOfSigmoid2 = log(1 - sigmoid(z));
%[ 1 x 1 ]= [1 x m] *[m x 1]
term2 = logOfSigmoid2' * (1 - y);
%theta without first term
theta_without_first_term = theta(2 : size_theta);
% theta square
square_theta = theta_without_first_term.^2;
%sum
sum_square_theta = sum(square_theta);
%[1 x 1] = (n+1) x 1
term3 = (lambda / ( 2 * m)) * sum_square_theta;
J = (1/m) * (-term1 - term2) + term3;



% error vector [m x 1]
error_vector = sigmoid(z) - y;
%gradient

%first term
%[1 x 1] = (1 x m) x (m x 1)
grad(1) = (1/m)* X(:,1)' * error_vector;

%other terms
% [(n) x 1] = [(n) x m]x [m x 1]
grad_term1 = (1/m)* X(:,2:n_plus_1)' * error_vector;

grad_term2 = lambda*(1/m)*theta(2: n_plus_1);
grad(2: n_plus_1) = grad_term1 + grad_term2;



% =============================================================

end
