from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = y.shape[0]
    for i in range(N):
        score_i = X[i] @ W
        nominator = np.exp(score_i - score_i.max())
        denominator = np.sum(nominator)
        prob_i = nominator / denominator
        prob_yi = prob_i[y[i]]
        loss_i = -np.log(prob_yi)
        
        partial_prob_score = -prob_yi*prob_i
        partial_prob_score[y[i]] += prob_yi
        
        dW_i = -X[i].reshape(-1, 1) @ partial_prob_score.reshape(1, -1) / prob_yi
        
        loss += loss_i
        dW += dW_i
        
    loss /= N
    loss += reg * np.sum(W**2)
    
    dW /= N
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = y.shape[0]
    S_ori = X @ W
    S = S_ori - S_ori.max(axis=1)[:, np.newaxis] # N,C
    
    correct_S = S[range(N), y]     # N,1
    exp = np.exp(S)                # N,C
    denom = np.sum(exp, axis=1)    # N,1
    log_sum = np.log(denom)        # N,1
    
    L = - np.sum(correct_S - log_sum) / N
    loss = L + reg * np.sum(W**2)
    
    # 计算dW
    prob = exp / denom[:, np.newaxis] # N,C
    
    mask_y = np.zeros(S.shape)       # N,C
    mask_y[np.arange(N), y] = 1
    
    mask_max = np.zeros(S.shape)
    mask_max[:,np.argmax(S_ori, axis=1)] = np.sum(prob - mask_y, axis=1)
    
    dW = mask_max + mask_y - prob



    # after this change, dW is partial of Loss over W
    dW = X.T @ dW
    dW = - dW / N + 2 * reg * W  
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
