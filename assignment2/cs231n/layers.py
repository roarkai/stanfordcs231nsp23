from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_2D = x.reshape((x.shape[0], -1))
    out = x_2D @ w + b
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    db = np.sum(dout, axis=0)
    
    x_2D = x.reshape((x.shape[0], -1))
    dw = x_2D.T @ dout
    
    dx = (dout @ w.T).reshape(x.shape)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = np.maximum(0, x)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = (x > 0) * dout

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = y.shape[0]
    
    # 调整x，避免overflow，对梯度没影响，相当于让exp每个元素都除常数exp(x.max)
    adj_x = x - np.max(x, axis=1).reshape(-1, 1)
    exp = np.exp(adj_x)
    normalization = np.sum(exp, axis=1).reshape(-1, 1)
    
    # 取correct_x直接计算loss，避免underflow
    adj_correct_score = adj_x[range(N), y]
    loss = -(np.sum(adj_correct_score) - np.sum(np.log(normalization))) / N
    
    # 计算dx
    d_correct_score = np.zeros(x.shape)
    d_correct_score[range(N), y] = 1
    dx = -exp / normalization + d_correct_score
    dx *= -1 / N

    ################################
    # rk's note:                   #
    #    下面这种方式会产生underflow  #
    ################################
    # exp = np.exp(x - np.max(x, axis=1).reshape(-1, 1)) # 对梯度没影响，相当于让exp各元素除常数exp(x.max)
    # prob = exp / exp.sum(axis=1).reshape(-1, 1)
    # prob_correct = prob[range(N), y]
    # loss = -np.sum(np.log(prob_correct)) / N
    
    # partial_prob_correct = -1 / prob_correct
    # partial_prob_over_score = -prob * prob_correct.reshape(-1, 1)
    # partial_prob_over_score[range(N), y] += prob_correct
    # dx = partial_prob_correct.reshape(-1, 1) * partial_prob_over_score / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****       
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.mean((x - batch_mean) ** 2, axis=0)
        batch_std_var = np.sqrt(batch_var + eps)
        normed_x = (x - batch_mean) / batch_std_var
        out = normed_x * gamma + beta
        
        # package information needed for backward in cache
        cache = (batch_std_var, normed_x, gamma)
        
        # compute running average mean and var for test time
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****      
        
        normed_x = (x - running_mean) / np.sqrt(running_var + eps)
        out = normed_x * gamma + beta
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ###########################################################################
    # norm = (x - mean) / sqrt(var + eps)                                     #
    ###########################################################################
    # d_L2x = partial_L2x           = dnorm * partial_norm2x
    #       + partial_L2var2x       + dnorm * partial_norm2var2x
    #       + partial_L2mean2x      + dnorm * partial_norm2mean2x
    # 
    # dnorm = gamma * dout
    # std_x = sqrt(var + eps)
    # 
    # (1)partial_L2x = 1 / sqrt(var + eps) * dnorm
    # 
    # (2)partial_L2mean2x:
    #  · d_meani2xi = 1 / N
    #  · partial_L2mean2x = 1 / (N * std_x) * np.sum(dnorm,axis=0)*np.ones(x.shape)
    #
    # (3)partial_L2var2x:
    #  · partial_vari2meani = 0
    #  · partial_vari2xi = 2 / N * (xi - meani)
    #  · d_vari2xi = partial_vari2meani * d_meani2xi + partial_vari2xi
    #              = partial_vari2xi
    #              = 2 / N * (xi - meani)
    #
    #  · partial_L2var = -0.5 * np.sum((x - mean) * dnorm, axis=0) * (var + eps)^(-3/2)
    # 
    #  · partial_L2var2x = partial_L2var * d_var2x
    #                    = -normed_x * (np.sum(normed_x*dnorm, axis=0) / (N * std_x)
    # 
    ###########################################################################    
    
    batch_std_var, normed_x, gamma = cache
    N = normed_x.shape[0]
    
    dnorm = gamma * dout
#     partial_L2x = 1 / std_var * dnorm
#     partial_L2mean2x = -np.sum(dnorm, axis=0) * np.ones(x.shape) / (N * std_var)
#     partial_L2var2x = -normed_x * np.sum(normed_x * dnorm, axis=0) / (N * std_var)
#     dx = partial_L2x + partial_L2mean2x + partial_L2var2x

    dctd1 = 1 / batch_std_var * dnorm
    dctd2 = -1 / N * normed_x * np.sum(normed_x*dnorm, axis=0) / batch_std_var
    dctd = dctd1 + dctd2
    dx = dctd - np.mean(dctd, axis=0)
    
    dgamma = np.sum(normed_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    batch_std_var, normed_x, gamma = cache
    N = normed_x.shape[0]
    
    dnorm = gamma * dout
    partial_1 = dnorm
    partial_2 = -np.sum(dnorm, axis=0) * np.ones(normed_x.shape) / N
    partial_3 = -normed_x * np.sum(normed_x * dnorm, axis=0) / N
    dx = (partial_1 + partial_2 + partial_3) / batch_std_var
    
    dgamma = np.sum(normed_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    mean = np.mean(x, axis=1).reshape(-1, 1)
    var = np.var(x, axis=1).reshape(-1, 1)
    std_var = np.sqrt(var + eps)
    normed_x = (x - mean) / std_var
    out = normed_x * gamma + beta
    cache = (x, normed_x, std_var, gamma)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x, normed_x, std_var, gamma = cache
    D = x.shape[1]  # 这里要用D，而不是N
    
    dnorm = gamma * dout
    partial_1 = dnorm
    partial_2 = -np.sum(dnorm, axis=1).reshape(-1, 1) * np.ones(x.shape) / D
    partial_3 = -normed_x * np.sum(normed_x * dnorm, axis=1).reshape(-1, 1) / D
    dx = (partial_1 + partial_2 + partial_3) / std_var    

    dgamma = np.sum(dout * normed_x, axis=0)
    dbeta = np.sum(dout, axis=0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = np.random.uniform(0, 1, size=x.shape)
        mask = (mask < p) / p
        out = mask * x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = mask * dout
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    P, S = conv_param['pad'], conv_param['stride']
    
    # x的pad方式：维度为N和C的那两个维度不pad，维度为H和W的两个维度前后pad的数量是P
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant', constant_values=0)
    _, _, Hpad, Wpad = x_pad.shape
    
    # check dimensions
    assert (Hpad - HH) % S == 0, "height does not work"
    assert (Wpad - WW) % S == 0, "width does not work"
    
    # init output
    Hout = (Hpad - HH) // S + 1
    Wout = (Wpad - WW) // S + 1
    out = np.zeros((N, F, Hout, Wout))
    
    # 将filters变成二维，每一维是一个filter, 每个filter的维度是：dim_f2D = C*HH*WW
    w_2D = w.reshape(F, -1).T
    
    # 遍历output的各个position，每次计算1个position的值
    for i in range(Hout):       # i是row index，j增加表示往下1行
        for j in range(Wout):   # j是column index, i增加表示往右1列         
            patch_ij = x_pad[:, :, i*S:(i*S+WW), j*S:(j*S+HH)].reshape(N, -1)
            out[:, :, i, j] = patch_ij @ w_2D + b
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    x, w, b, conv_param = cache

    # 取各维度的值
    P, S = conv_param['pad'], conv_param['stride']
    N, F, Hout, Wout = dout.shape
    _, C, H, W = x.shape
    _, _, HH, WW = w.shape
    
    # bias的shape是(F,)
    db = np.sum(dout, axis=(0, 2, 3))
    
    # x的pad方式：维度为N和C的那两个维度不pad，维度为H和W的两个维度前后pad的数量是P
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant', constant_values=0)

    # 初始化梯度值为0
    dx = np.zeros(x.shape)            # x shape: N, C, H, W
    dw = np.zeros(w.shape)            # w shape: F, C, HH, WW
    dx_pad = np.zeros((x_pad.shape))   # x_pad shape: N, C, (H+2P), (W+2P)
                      
    # 将filters变成二维，每一维是一个filter, 每个filter的维度是：dim_f2D = C*HH*WW
    # w_2D.shape = (F, dim_f2D)
    w_2D = w.reshape(F, -1)

    # 遍历output的各个position，每次计算1个position相对w的梯度，累加
    for i in range(Hout):       # i是row index，j增加表示往下1行
        for j in range(Wout):   # j是column index, i增加表示往右1列
            patch_ij = x_pad[:, :, i*S:(i*S+WW), j*S:(j*S+HH)].reshape(N, -1)
            dout_ij = dout[:, :, i, j]
            dw += (dout_ij.T @ patch_ij).reshape(F, C, HH, WW)
            dx_pad[:, :, i*S:(i*S+WW), j*S:(j*S+HH)] += (dout_ij @ w_2D).reshape(N, C, HH, WW)
    
    dx = dx_pad[:, :, P:(P+H), P:(P+W)]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    Hpool = pool_param['pool_height']
    Wpool = pool_param['pool_width']
    S = pool_param['stride']

    # check dimensions
    assert Hpool == Wpool, "invalid pool params"
    assert H % Hpool == 0, "height does not work"
    assert W % Wpool == 0, "width does not work"
    
    Hout = 1 + (H - Hpool) // S  # column number
    Wout = 1 + (W - Wpool) // S  # row number
    
    out = np.zeros((N, C, Hout, Wout))
    for i in range(Hout):        # i是row index，j增加表示往下1行
        for j in range(Wout):    # j是column index, i增加表示往右1列
            patch_ij = x[:, :, i*S:(i*S+Wpool), j*S:(j*S+Hpool)]
            out[:, :, i, j] = np.max(patch_ij, axis=(2, 3))
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param  = cache
    Hpool = pool_param['pool_height']
    Wpool = pool_param['pool_width']
    S = pool_param['stride']
    N, C, Hout, Wout = dout.shape
    _, _, H, W = x.shape
    
    dx = np.zeros(x.shape)
    for i in range(Hout):        # i是row index，j增加表示往下1行
        for j in range(Wout):    # j是column index, i增加表示往右1列
            [n, c] = np.indices((N, C))
            # 为符合np.argmax的参数规则，改变patch形状
            patch_ij = x[:, :, i*S:(i*S+Hpool), j*S:(j*S+Wpool)].reshape(N, C, -1)
            k, l = np.unravel_index(np.argmax(patch_ij, axis=-1), (Hpool, Wpool))
            dx[n, c, i*S+k, j*S+l] += dout[n, c, i, j]

    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    pass

    ###########################################################################
    # rk's note: unnecessary way                                      #
    ###########################################################################    
#     N, C, H, W = x.shape
    
#     # get batchnorm params
#     mode = bn_param['mode']
    
#     # 注意，开始训练时bn_param没有'running_mean'和'running_var', 因此要用get method家default value
#     eps = bn_param.get('eps', 1e-5)
#     m = bn_param.get('momentum', 0.9)
#     running_mean = bn_param.get('running_mean', np.zeros((1, C, 1, 1), dtype=x.dtype)) 
#     running_var = bn_param.get('running_var', np.zeros((1, C, 1, 1), dtype=x.dtype))

#     if mode == 'train':
#         # compute mean and var
#         mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
#         var = np.var(x, axis=(0, 2, 3), keepdims=True)
#         std_var = np.sqrt(var + eps)
    
#         # normalize
#         normed_x = (x - mean) / std_var
#         out = normed_x * gamma.reshape((1, C, 1, 1)) + beta.reshape((1, C, 1, 1))
        
#         # update running_mean and running_var
#         running_mean = m * running_mean + (1 - m) * mean
#         running_var = m * running_var.reshape(1, C, 1, 1) + (1 - m) * var
        
#         # package values needed in backward into cache
#         cache = (x, normed_x, std_var, bn_param)
        
#     elif mode == 'test':
#         normed_x = (x - running_mean) / running_var
#         out = normed_x * gamma.reshape((1, C, 1, 1)) + beta.reshape((1, C, 1, 1))
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****   
    
    N, C, H, W = x.shape
    M = C // G
    x = x.reshape(N, G, M, H, W)
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    std_var = np.sqrt(var + eps)
    normed_x = (x - mean) / std_var

    gamma = np.tile(gamma.reshape(1, G, M, 1, 1), (N, 1, 1, 1, 1))  
    beta = np.tile(beta.reshape(1, G, M, 1, 1), (N, 1, 1, 1, 1))
    out = normed_x * gamma + beta
    
    out = out.reshape(N, C, H, W)
    cache = (x, normed_x, gamma, std_var, G)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 注意cache中的变量形态是在forward中改变后的状态：
    # x.shape = (N, G, M, H, W)
    # normed_x.shape = (N, G, M, H, W)
    # gamma.shape = (N, G, M, 1, 1)
    # std_var.shape = (N, G, 1, 1, 1)
    
    x, normed_x, gamma, std_var, G = cache
    N, C, H, W = dout.shape
    M = C // G
    
    # 
    dout = dout.reshape((N, G, M, H, W))
    dnormed_x = dout * gamma
    partial_1 = dnormed_x
    partial_2 = -np.sum(dnormed_x, axis=(2, 3, 4), keepdims=True) * np.ones(x.shape) / (M * H * W)
    partial_3 = -normed_x * np.sum(normed_x * dnormed_x, axis=(2, 3, 4), keepdims=True) / (M * H * W)
    dx = ((partial_1 + partial_2 + partial_3) / std_var).reshape((N, C, H, W))
       
    dgamma = np.sum(dout * normed_x, axis=(0, 3, 4)).reshape((1, C, 1, 1))
    dbeta = np.sum(dout, axis=(0, 3, 4)).reshape((1, C, 1, 1))
    
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
