function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% Refer to the Programming tutorial from the Resources Menu for Tom
% Mosher's guidelines on how to proceed.
% copy-pasted as comments here


eye_matrix = eye(num_labels);
% ****create a matrix for y
y_matrix = eye_matrix(y,:)';
%**Forward propagation from previous programming exercise
% Add ones to the X data matrix
X = [ones(m, 1) X];
% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
%HIDDEN LAYER
% [(sj+1) x m ]
a_one = X';

% z2 equals the product of a1 and ?1
%[s(j+1) x m ] = [s(j+1) x (sj +1)  (sj+1) x m]
z_two = Theta1 * a_one;

% a2 is the result of passing z2 through g()
%[s(j+1) x m ]
a_two = sigmoid(z_two);
% % transpose so that the bias units are the first column
a_two = a_two';

% Then add a column of bias units to a2 (as the first column)
% Add ones to the a_two matrix
a_two = [ones(m, 1) a_two];



% NOTE: Be sure you DON'T add the bias units as a new row of Theta.
% 
% z3 equals the product of a2 and ?2
%OUTPUT LAYER
% [num_labels x m ] = [num_labels x (s(j+1)+1)  [s(j+1)+1 x m ]  ]
z_three = Theta2 * a_two';

% a3 is the result of passing z3 through g()
a_three = sigmoid(z_three);

%--------------------------------------------------------------- 
% Cost Function, non-regularized: 
% 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% using a3, your y_matrix, and m (the number of training examples). 
% Note that the 'h' argument inside the log() function is exactly a3.
% Cost should be a scalar value.
% Since y_matrix and a3 are both matrices,
% you need to compute the double-sum. 

% for loop over num_labels
for k = 1:num_labels
term1 =  y_matrix(k,:) .* log( a_three(k,:) );
term2 = ( 1 - y_matrix(k,:) ).* log( 1 - a_three(k,:) );
J = J + ( 1 / m )* ( - term1 - term2);
end
J = sum(J(:));

% Cost Regularization:
% 4 - Compute the regularized component of the cost according to ex4.pdf Page 6,
% using Theta1 and Theta2 (excluding the Theta columns for the bias units i.e the first columns of each of these matrices),
% along with lambda, and m. The easiest method to do this is to compute the regularization terms separately,
% then add them to the unregularized cost from Step 3.
% 
% You can run ex4.m to check the regularized cost, then you can submit this portion to the grader.
%25 x 400
%remove first column which contains biased units
Theta1_no_bias = Theta1( :, 2 : end);

%10 x 25
%remove first column which contains biased units
Theta2_no_bias = Theta2( :, 2 : end );
regularization_param = 0;
%%%% for loop over num_labels
%square
sq = Theta1_no_bias.^2;
term1 = sum(sq(:));

sq1 = Theta2_no_bias.^2;
term2 = sum(sq1(:));

term3 = term1 + term2;
regularization_param = regularization_param + 0.5*lambda*(1/m)*term3;


% add regularization parameter to cost function J
J = J + regularization_param; 

% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.(**not recommended by Tom)


% 1: Perform forward propagation, see the separate tutorial if necessary.
% 2: ?3 or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
d3 = a_three - y_matrix;

% % transpose 
% d3 = d3';
% 3: z2 came from the forward propagation process - it's the product of a1 and Theta1, prior to applying the sigmoid() function.
%Dimensions are (m x n) ? (n x h) --> (m x h)
% 4: ?2 or d2 is tricky. It uses the (:,2:end) columns of Theta2.
%d2 is the product of d3 and Theta2(no bias), 
%then element-wise scaled by sigmoid gradient of z2.
%The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
d2 = (Theta2_no_bias' * d3 ).* sigmoidGradient(z_two);
%transpose
% d2= d2';
% 5: ?1 or Delta1 is the product of d2 and a1. The size is (h x m) ? (m x n) --> (h x n)
Delta1 = d2 * a_one';% transpose so that the bias units are the first column

% 6: ?2 or Delta2 is the product of d3 and a2. The size is (r x m) ? (m x [h+1]) --> (r x [h+1])
Delta2 = d3 * a_two;

% 7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta1_grad = (1/m) *Delta1;
Theta2_grad = (1/m) *Delta2;
% Now you have the unregularized gradients. Check your results using ex4.m, and submit this portion to the grader.

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
