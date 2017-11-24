# -*- coding: utf-8 -*-
###############################################################################
#
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
home work result.

Authors: wangxiao05(wangxiao05@baidu.com)
Date:    
"""

import numpy as np
from random import shuffle
# from past.builtins import xrange


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
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

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
    pass
    scores = np.dot(X, W)
    num_train = X.shape[0]
    # Shift the score to deal with exponantial instability
    scores -= np.max(scores, axis=1).reshape(-1, 1)
    exp_scores = np.exp(scores)
    exp_norms = np.sum(exp_scores, axis=1)
    p = exp_scores / (exp_norms.reshape(-1, 1) + 0.0000001)
    loss = (np.sum(- np.log(p[range(0, num_train), y]))) / num_train + 0.5 * reg * np.sum(W * W)
    # compute gradient of log likelihood loss
    sg = np.zeros_like(p)
    sg[range(0, num_train), y] = 1.0
    dW = np.dot(X.T, (p - sg)) / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
