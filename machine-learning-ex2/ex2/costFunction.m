function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%[ m  x 1 ] = [m x (N + 1)] x [(N+1) x 1]
z = X * theta;
%[ m x 1 ]
logOfSigmoid =  log(sigmoid(z));
%[ 1 x 1 ] = [ 1 x m ] x [m x 1]
term1 = logOfSigmoid' * y;
%[ m x 1]
logOfSigmoid2 = log(1 - sigmoid(z));
%[ 1 x 1 ]= [1 x m] *[m x 1]
term2 = logOfSigmoid2' * (1 - y);
J = (1/m) * (-term1 - term2);


% error vector [m x 1]
error_vector = sigmoid(z) - y;
%gradient
% [(N+1) x 1] = [(N + 1) x m]x [m x 1]
grad = (1/m)* X' * error_vector;

% =============================================================

end
