import numpy as np
from Utils.data_utils import plot_conv_images

def conv_forward(x, w, b, conv_param):
    """
    Computes the forward pass for a convolutional layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - w: Weights, of shape (F, WH, WW, C)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields, of shape (1, SH, SW, 1)
      - 'padding': "valid" or "same". "valid" means no padding.
        "same" means zero-padding the input so that the output has the shape as (N, ceil(H / SH), ceil(W / SW), F)
        If the padding on both sides (top vs bottom, left vs right) are off by one, the bottom and right get the additional padding.
         
    Outputs:
    - out: Output data
    - cache: (x, w, b, conv_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, w, b, conv_param)
    return out, cache
    

def conv_backward(dout, cache):
    """
    Computes the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Outputs:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx, dw, db

def max_pool_forward(x, pool_param):
    """
    Computes the forward pass for a pooling layer.
    
    For your convenience, you only have to implement padding=valid.
    
    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The number of pixels between adjacent pooling regions, of shape (1, SH, SW, 1)

    Outputs:
    - out: Output data
    - cache: (x, pool_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Computes the backward pass for a max pooling layer.

    For your convenience, you only have to implement padding=valid.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in max_pool_forward.

    Outputs:
    - dx: Gradient with respect to x
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx

def _rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def Test_conv_forward(num):
    """ Test conv_forward function """
    if num == 1:
        x_shape = (2, 4, 8, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[  5.12264676e-02,  -7.46786231e-02],
                                  [ -1.46819650e-03,   4.58694441e-02]],
                                 [[ -2.29811741e-01,   5.68244402e-01],
                                  [ -2.82506405e-01,   6.88792470e-01]]],
                                [[[ -5.10849950e-01,   1.21116743e+00],
                                  [ -5.63544614e-01,   1.33171550e+00]],
                                 [[ -7.91888159e-01,   1.85409045e+00],
                                  [ -8.44582823e-01,   1.97463852e+00]]]])
    else:
        x_shape = (2, 5, 5, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[ -5.28344995e-04,  -9.72797373e-02],
                                  [  2.48150793e-02,  -4.31486506e-02],
                                  [ -4.44809367e-02,   3.35499072e-02]],
                                 [[ -2.01784949e-01,   5.34249607e-01],
                                  [ -3.12925889e-01,   7.29491646e-01],
                                  [ -2.82750250e-01,   3.50471227e-01]]],
                                [[[ -3.35956019e-01,   9.55269170e-01],
                                  [ -5.38086534e-01,   1.24458518e+00],
                                  [ -4.41596459e-01,   5.61752106e-01]],                             
                                 [[ -5.37212623e-01,   1.58679851e+00],
                                  [ -8.75827502e-01,   2.01722547e+00],
                                  [ -6.79865772e-01,   8.78673426e-01]]]])
        
    return _rel_error(out, correct_out)


def Test_conv_forward_IP(x):
    """ Test conv_forward function with image processing """
    w = np.zeros((2, 3, 3, 3))
    w[0, 1, 1, :] = [0.3, 0.6, 0.1]
    w[1, :, :, 2] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    b = np.array([0, 128])
    
    out, _ = conv_forward(x, w, b, {'stride': np.array([1,1,1,1]), 'padding': 'same'})
    plot_conv_images(x, out)
    return
    
def Test_max_pool_forward():   
    """ Test max_pool_forward function """
    x_shape = (2, 5, 5, 3)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    out, _ = max_pool_forward(x, pool_param)
    correct_out = np.array([[[[ 0.03288591,  0.03691275,  0.0409396 ]],
                             [[ 0.15369128,  0.15771812,  0.16174497]]],
                            [[[ 0.33489933,  0.33892617,  0.34295302]],
                             [[ 0.4557047,   0.45973154,  0.46375839]]]])
    return _rel_error(out, correct_out)

def _eval_numerical_gradient_array(f, x, df, h=1e-5):
    """ Evaluate a numeric gradient for a function """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        p = np.array(x)
        p[ix] = x[ix] + h
        pos = f(p)
        p[ix] = x[ix] - h
        neg = f(p)
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def Test_conv_backward(num):
    """ Test conv_backward function """
    if num == 1:
        x = np.random.randn(2, 4, 8, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        dout = np.random.randn(2, 2, 2, 2)
    else:
        x = np.random.randn(2, 5, 5, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        dout = np.random.randn(2, 2, 3, 2)
    
    out, cache = conv_forward(x, w, b, conv_param)
    dx, dw, db = conv_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = _eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = _eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)
    
    return (_rel_error(dx, dx_num), _rel_error(dw, dw_num), _rel_error(db, db_num))

def Test_max_pool_backward():
    """ Test max_pool_backward function """
    x = np.random.randn(2, 5, 5, 3)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    dout = np.random.randn(2, 2, 1, 3)
    
    out, cache = max_pool_forward(x, pool_param)
    dx = max_pool_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)
    
    return _rel_error(dx, dx_num)