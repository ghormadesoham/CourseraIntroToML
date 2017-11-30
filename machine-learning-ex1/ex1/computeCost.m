function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% temp = 0;
%  theta0=theta(1);
%  theta1= theta(2);
%Method 1-iterative  
% for i =1 : m
%     temp = temp + (1/(2*m))*(theta0 + theta1 *X(i,2 )- y(i))^2; 
% end
% method 2 vectorized i.e using matrices
temp1 =theta'* X';
temp2 = temp1';
temp3 = 0.5*(1/m)*(temp2 - y ).^2;
   J = sum(temp3);    
%J = temp;
%  J = 0.5 *(1/m) * (theta'* X(2,2) - y)^2;
% =========================================================================

end
