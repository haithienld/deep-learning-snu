import numpy as np
from Utils.data_utils import plot_conv_images
import math 

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
    N,H,W,C =x.shape #batch-size (number of images - one times), Height, Width, Channel
    # print("Height, Width, Channel of x:Image",H,W,C)
    F,WH,WW,C = w.shape #Number of Filters, Filter Height, Filter Width, Channel
    # print("Height, Width, Channel of w:filter", WH,WW,C)
    S = conv_param['stride']
    SH = S[1]
    SW = S[2]
    # print("Height, Width of stride", SH,SW)
    if (conv_param['padding'] =='valid'):
        Ph = Pw = 0
        H_out = int((H-WH)/SH) +1
        W_out = int((W-WW)/SW) +1
        # x_pad = np.pad(x, ((0,), (Ph,), (Pw,), (0,)), mode='constant')
    if (conv_param['padding'] =='same'):
        # H_out = int((H-WH+2*Ph)/SH) +1
        # W_out = int((W-WW+2*Pw)/SW) +1
        H_out = math.ceil(H/SH)
        W_out = math.ceil(W/SW)
        Ph = int((H_out-1)*SH + WH - H)
        Pw = int((W_out-1)*SW + WW - W)
    #If padding == same ==> Calculate PH_Top, Ph_Bottom, Pw_Left, Pw_Right
    Pht = int(Ph/2)
    Phb = Ph - Pht
    Pwl = int(Pw/2)
    Pwr = Pw - Pwl
    # x_pad = np.pad(x, ((0,), (Pht,Phb), (Pwr,Pwl), (0,)), mode='constant')
    x_pad = np.zeros((N, H + Pht + Phb, W + Pwl + Pwr, C))
    x_pad[:, Pht:H+Pht, Pwl:W+Pwl, :] = x
        #print(x_pad)
        # Shape N,F
    out = np.zeros((N, H_out, W_out, F))
    for i in range(N):
        image = x_pad[ i, :, : , : ]
        for j in range(H_out):
            for k in range(W_out):
                for l in range(F):
                    h1 = j*SH
                    h2 = j*SH + WH
                    w1 = k*SW
                    w2 = k*SW + WW
                    image_temp = image[h1:h2, w1:w2,:]
                    out[i,j,k,l] = np.sum(np.multiply(image_temp, w[l, :, :, :])) + b[l]
    #a output data has size H_out, W_out
    # H_out = int((H-WH+2*Ph)/SH) +1
    # W_out = int((W-WW+2*Pw)/SW) +1

    # for i in range(C):
    #     for j in range (H + 2 * Ph):
    #         for k in range (W + 2 * Pw):
    #             a = 3

    
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
    (x, w, b, conv_param) = cache
    N,H,W,C = x.shape
    F,WH,WW,C = w.shape
    S = conv_param['stride']
    SH = S[1]
    SW = S[2]
    padding = conv_param['padding']
    if padding == 'valid':
        Ph = Pw = 0
        H_out = int((H - WH + 2*Ph)/SH + 1)
        W_out = int((W - WW + 2*Pw)/SW + 1)
    if padding == 'same':
        H_out = math.ceil(H/SH)
        W_out = math.ceil(W/SW)
        print("Height output, Width output of Padding = Same", H_out,W_out)
        Ph = int((H_out-1)*SH + WH - H)
        Pw = int((W_out-1)*SW + WW - W)
    #If padding == same ==> Calculate PH_Top, Ph_Bottom, Pw_Left, Pw_Right
    Pht = int(Ph/2)
    Phb = Ph - Pht
    Pwl = int(Pw/2)
    Pwr = Pw - Pwl
    x_pad = np.zeros((N,H + Pht + Phb, W + Pwl + Pwr,C))
    x_pad[:,Pht:H + Pht, Pwl:W + Pwl,:] = x
    dw = np.zeros(w.shape)
    for i in range(N):
        for j in range(H_out):
            for k in range(W_out):
                for l in range(F):
                    dw[l,:,:,:] += x_pad[l,j*SH:j*SH + WH, k*SW:k*SW+WW,:]*dout[i,j,k,l]
    db = np.zeros(b.shape)
    for f in range(F):
        db[f] += np.sum(dout[:,:,:,f])
    dx = np.zeros(x.shape)
    dx_pad = np.zeros(x_pad.shape)
    for i in range(N):
        for j in range(H_out):
            for k in range(W_out):
                for l in range(F):
                    for c in range(C):
                        dx_pad[i,j*SH:j*SH + WH, k*SW:k*SW+WW,c] += w[l,:,:,c]*dout[i,j,k,l]
    dx = dx_pad[:,Pht:H+Pht, Pwl:W+Pwl, :]
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx, dw, db

def max_pool_forward(x, pool_param):
    #https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
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
    N,H,W,C = x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    S = pool_param['stride']
    SH = S[1]
    SW = S[2]
    #if padding == 'valid'
    Ph = Pw = 0
    H_out = int((H - pool_H + 2*Ph)/SH + 1)
    W_out = int((W - pool_W + 2*Pw)/SW + 1)
    x_pad = np.zeros((N,H,W,C))
    x_pad[:,0:H,0:W,:] = x
    out = np.zeros((N, H_out, W_out, C))
    for i in range(N):
        image = x_pad[ i, :, : , : ]
        for j in range(H_out):
            for k in range(W_out):
                for l in range(C):
                    h1 = j*SH
                    h2 = j*SH + pool_H
                    w1 = k*SW
                    w2 = k*SW + pool_W
                    out[i,j,k,l] = np.amax(x_pad[i,h1:h2,w1:w2,l])
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
    (x,pool_param) = cache
    N,H,W,C = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    S = pool_param['stride']
    SH, SW = S[1], S[2]
    # padding == 'valid'
    PH = 0
    PW = 0
    # output size
    H_out = int((H - pool_height + 2*0)/SH + 1)
    W_out = int((W - pool_width + 2*0)/SW + 1)
    dx = np.zeros(x.shape)
    #print(out.shape)
    for i in range(N):
        for j in range(H_out):
            for k in range(W_out):
                for l in range(C):
                    temp = x[i, j*SH:j*SH+pool_height, k*SW:k*SW+pool_width, l]
                    max_idx = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                    temp = np.zeros(temp.shape)
                    temp[max_idx] = 1
                    dx[i, j*SH:j*SH+pool_height, k*SW:k*SW+pool_width, l] += temp * dout[i, j, k, l]
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