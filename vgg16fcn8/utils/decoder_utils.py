from __future__ import division

import tensorflow as tf
import classes_utils as cutils

def evaluation_original(hyp, images, labels, decoded_logits, losses, global_step):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    eval_list = []
    num_classes = cutils.get_num_classes(hyp)
    logits = tf.reshape(decoded_logits['logits'], (-1, num_classes))
    labels = tf.reshape(labels, (-1, num_classes))
    pred = tf.argmax(logits, dimension=1)
    tp = 0
    all_fn = 0
    for i in range(num_classes):
        negativ = tf.to_int32(tf.equal(pred, i))
        for j in range(num_classes):
            all_fn += tf.reduce_sum(negativ * labels[:, j])
            if i is j:
                tp += tf.reduce_sum(negativ * labels[:, j])

    eval_list.append(('Acc. ', (tp / all_fn)))
    eval_list.append(('xentropy', losses['xentropy']))
    eval_list.append(('weight_loss', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list


def evaluation_jaccard(hyp, images, labels, decoded_logits, losses, global_step):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    eval_list = []
    num_classes = hyp['arch']['num_classes']
    logits = tf.reshape(decoded_logits['logits'], (-1, num_classes))
    labels = tf.reshape(labels, (-1, num_classes))
    pred = tf.argmax(logits, dimension=1)
    j = 0
    for i in range(num_classes):
        negativ = tf.to_int32(tf.equal(pred, i))
        tp = tf.reduce_sum(negativ * labels[:, i])
        total_prediction = tf.reduce_sum(negativ)
        total_label = tf.reduce_sum(labels[:, i])
        jnum = tf.where(tf.logical_and(tf.equal(total_label, 0), tf.equal(total_prediction, 0)), tf.constant(1), tp)
        jden = tf.where(tf.logical_and(tf.equal(total_label, 0), tf.equal(total_prediction, 0)), tf.constant(1), total_prediction + total_label - tp)
        j += jnum / jden

    eval_list.append(('Acc. ', (j/num_classes)))
    eval_list.append(('xentropy', losses['xentropy']))
    eval_list.append(('weight_loss', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list