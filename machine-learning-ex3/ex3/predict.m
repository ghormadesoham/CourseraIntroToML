function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% Add ones to the X data matrix
X = [ones(m, 1) X];
%HIDDEN LAYER
% [(sj+1) x m ]
a_one = X';

%[s(j+1) x m ] = [s(j+1) x (sj +1)  (sj+1) x m]
z_two = Theta1 * a_one;

%[s(j+1) x m ]
a_two = sigmoid(z_two);
% take a transpose
a_two = a_two';
% Add ones to the a_two matrix
a_two = [ones(m, 1) a_two];
% take a transpose again 
%[s(j+1)+1 x m ]
a_two = a_two';


%OUTPUT LAYER
% [num_labels x m ] = [num_labels x (s(j+1)+1)  [s(j+1)+1 x m ]  ]
z_three = Theta2 * a_two;
a_three = sigmoid(z_three);

% max computation
[M,p] = max(a_three,[],1);
% =========================================================================

% take a transpose
p = p';
end
