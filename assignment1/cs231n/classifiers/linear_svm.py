import numpy as np
from random import shuffle
# from past.builtins import xrange
import time

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -1.0 * X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.

    loss /= num_train
    dW /= num_train
    # AFTER THE BATCH

    dW += 2 * reg * W
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass
    tic = time.time()
    num_train = y.shape[0]
    scores = X.dot(W)
    correct_class_score = scores[range(0, num_train), y]
    margins = (scores - correct_class_score[:, np.newaxis] + 1)
    margins[range(0, num_train), y] = 0
    margins[margins < 0] = 0
    toc = time.time()
    print toc - tic
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)
    mask_mat = np.zeros(margins.shape)
    mask_mat[margins > 0] = 1
    toc = time.time()
    print 'mask_mat[margins > 0] = 1', toc - tic
    cube = X[:, :, np.newaxis] * mask_mat[:, np.newaxis, :]
    toc = time.time()
    print 'cube =', toc - tic
    correct_class_gradients = -1.0 * np.sum(cube, axis=2)
    toc = time.time()
    print 'correct_class_gradients =', toc - tic
    cube[range(0, num_train), :, y] += correct_class_gradients
    toc = time.time()
    print 'cube[range(0, num_train), :, y]:', toc - tic
    dW = np.sum(cube, axis=0)
    print dW[range(0, num_train), y].shape
    print correct_class_gradients.shape
    toc = time.time()
    print 'dW = np.sum(cube, axis=0):', toc - tic
    dW /= num_train
    dW += 2 * reg * W
    toc = time.time()
    print toc - tic
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized_1(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass
    tic = time.time()
    num_train = y.shape[0]
    scores = X.dot(W)
    correct_class_score = scores[range(0, num_train), y]
    margins = (scores - correct_class_score[:, np.newaxis] + 1)
    margins[range(0, num_train), y] = 0
    margins[margins < 0] = 0
    toc = time.time()
    print toc - tic
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)
    mask_mat = np.zeros(margins.shape)
    mask_mat[margins > 0] = 1
    toc = time.time()
    print 'mask_mat[margins > 0] = 1', toc - tic
    cube = X[:, :, np.newaxis] * mask_mat[:, np.newaxis, :]
    toc = time.time()
    print 'cube =', toc - tic
    correct_class_gradients = -1.0 * np.sum(cube, axis=2)
    toc = time.time()
    print 'correct_class_gradients =', toc - tic
    cube[range(0, num_train), :, y] += correct_class_gradients
    toc = time.time()
    print 'cube[range(0, num_train), :, y]:', toc - tic
    dW = np.sum(cube, axis=0)
    print dW[range(0, num_train), y].shape
    print correct_class_gradients.shape
    toc = time.time()
    print 'dW = np.sum(cube, axis=0):', toc - tic
    dW /= num_train
    dW += 2 * reg * W
    toc = time.time()
    print toc - tic
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
