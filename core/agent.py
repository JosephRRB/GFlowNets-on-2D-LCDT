import dgl
import dgl.function as fn
import tensorflow as tf


def _construct_segment_pair_auxillary_graph(original_graph, segment_pair):
    seg_a_pt0 = segment_pair[:, 0]
    seg_a_pt1 = segment_pair[:, 1]
    seg_b_pt0 = segment_pair[:, 2]
    seg_b_pt1 = segment_pair[:, 3]

    n_seg_pairs = segment_pair.shape[0]
    aux_join_pt0 = tf.range(0, n_seg_pairs, dtype=tf.int64)
    aux_join_pt1 = tf.range(n_seg_pairs, 2 * n_seg_pairs, dtype=tf.int64)
    aux_join_pairs = tf.range(0, n_seg_pairs, dtype=tf.int64)

    seg_inds, pt_inds = original_graph.edges(etype=('segment', 'has', 'point'))
    aux_graph = dgl.heterograph({
        ('point', 'in', 'segment'): (pt_inds, seg_inds),
        ('point', 'join', 'aux_pt_pair'): (
            tf.concat([seg_a_pt0, seg_a_pt1, seg_b_pt0, seg_b_pt1], axis=0),
            tf.concat([aux_join_pt0, aux_join_pt1, aux_join_pt0, aux_join_pt1],
                      axis=0)
        ),
        ('aux_pt_pair', 'join', 'seg_pairs'): (
            tf.concat([aux_join_pt0, aux_join_pt1], axis=0),
            tf.concat([aux_join_pairs, aux_join_pairs], axis=0)
        )
    })  # .long()
    aux_graph.edges[('point', 'join', 'aux_pt_pair')].data['weight'] = (
        tf.concat([
            tf.ones(shape=(2 * n_seg_pairs)), -tf.ones(shape=(2 * n_seg_pairs))
        ], axis=0)
    )
    return aux_graph


def _calculate_segment_pair_probabilities(aux_graph, point_logits):
    with aux_graph.local_scope():
        aux_graph.nodes['point'].data['logit'] = point_logits

        aux_graph.update_all(
            message_func=fn.u_mul_e('logit', 'weight', 'm'),
            reduce_func=fn.sum('m', 'pt_pair_logit'),
            etype=('point', 'join', 'aux_pt_pair')
        )
        aux_graph.nodes['aux_pt_pair'].data['pt_pair_logit'] = tf.math.tanh(
            aux_graph.nodes['aux_pt_pair'].data['pt_pair_logit']
        )
        aux_graph.update_all(
            message_func=fn.copy_u('pt_pair_logit', 'm'),
            reduce_func=fn.sum('m', 'seg_pair_logit'),
            etype=('aux_pt_pair', 'join', 'seg_pairs')
        )
        aux_graph.nodes['seg_pairs'].data['seg_pair_logit'] = -tf.math.abs(
            aux_graph.nodes['seg_pairs'].data['seg_pair_logit']
        )
        probabilities = dgl.softmax_nodes(
            aux_graph, 'seg_pair_logit', ntype='seg_pairs'
        )
        return probabilities