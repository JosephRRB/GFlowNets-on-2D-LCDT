import dgl
import dgl.function as fn
import tensorflow as tf

from core.agent.endpoint_pair_combinations import extract_endpoint_pair_combinations
from core.agent.policy_network import HeteroGraphPolicyNetwork


AGENT_HYPERPARAMETERS = {
    "policy_hyperparams": {
        "n_tri_feats": 1,
        "n_seg_feats": 2,
        "n_angle_feats": 5,
        "n_pt_feats": 5,
        "n_local_hidden_nodes_1": 16,
        "n_local_hidden_nodes_2": 8,
        "n_global_hidden_nodes_1": 32,
        "n_global_hidden_nodes_2": 16,
    }
}


class Agent:
    def __init__(self, hyperparamerters=None):

        if hyperparamerters is None:
            hyperparamerters = AGENT_HYPERPARAMETERS
        self.policy_network = HeteroGraphPolicyNetwork(
            **hyperparamerters["policy_hyperparams"]
        )


    def act(self, triangulation):
        """
        should only accept one connected triangulation (not batch)
        returned pt indices should be used for current triangulation
        """
        endpt_combos = extract_endpoint_pair_combinations(triangulation)
        point_logits, triangulation_logits = self.policy_network(triangulation)

        chosen_combo_type, encoded_action_type = _choose_main_action_type(
            triangulation_logits, endpt_combos
        )
        aux_graph = _choose_segment_pair_to_join(
            triangulation, chosen_combo_type, point_logits
        )

        agent_terminal_action = _choose_terminal_action(triangulation_logits)




        # main_action_and_terminate = tf.concat(
        #     [encoded_action_type, terminal_action], axis=0
        # )

        ## terminate in env.step if no more boundaries?
        # return aux_graph, main_action_and_terminate


def _choose_segment_pair_to_join(triangulation, chosen_combos, point_logits):
    aux_graph = _construct_segment_pair_auxillary_graph(
        triangulation, chosen_combos
    )
    pairs_to_join_probas = _calculate_segment_pair_probabilities(
        aux_graph, point_logits
    )
    chosen_pair_index = tf.random.categorical(pairs_to_join_probas, 1)

    endpoints_to_join = chosen_combos[chosen_pair_index[0, 0], :]
    segments_to_join = _find_segment_from_endpoints(
        aux_graph, endpoints_to_join[:2]
    )


    chosen_seg_pair = tf.scatter_nd(
        chosen_pair_index,
        tf.constant([1]),
        shape=(pairs_to_join_probas.shape[1],),
    )
    aux_graph.nodes["seg_pairs"].data["chosen_segment_pair"] = chosen_seg_pair
    return aux_graph


def _find_segment_from_endpoints(aux_graph, endpoints):
    segments_of_pt0 = aux_graph.successors(
        endpoints[0], etype=('point', 'in', 'segment')
    )
    segments_of_pt1 = aux_graph.successors(
        endpoints[1], etype=('point', 'in', 'segment')
    )
    common_segment = tf.sets.intersection(
        tf.expand_dims(segments_of_pt0, 0), tf.expand_dims(segments_of_pt1, 0)
    ).values
    return common_segment

def _choose_main_action_type(triangulation_logits, endpt_combos):
    available_combos = [t.shape[0] != 0 for t in endpt_combos]
    masked_logits = tf.where(
        available_combos, triangulation_logits[:, :-1], tf.float32.min
    )
    action_type = tf.random.categorical(masked_logits, 1)[0, 0]
    chosen_combos = endpt_combos[action_type]

    masked_action_type = tf.constant(available_combos, dtype=tf.float32) - 1
    encoded_action_type = masked_action_type + tf.one_hot(action_type, 6)
    return chosen_combos, encoded_action_type


def _choose_terminal_action(triangulation_logits):
    terminal_prob = tf.nn.sigmoid(triangulation_logits[0, -1:])
    terminal_action = tf.cast(
        tf.math.less(
            tf.random.uniform(shape=(1,), minval=0, maxval=1),
            terminal_prob,
        ),
        dtype=tf.float32,
    )
    return terminal_action


def _construct_segment_pair_auxillary_graph(original_graph, segment_pair):
    n_seg_pairs = segment_pair.shape[0]
    aux_join_pt0 = tf.range(0, n_seg_pairs, dtype=tf.int64)
    aux_join_pt1 = tf.range(n_seg_pairs, 2 * n_seg_pairs, dtype=tf.int64)
    aux_join_pairs = tf.range(0, n_seg_pairs, dtype=tf.int64)

    pt_indices = tf.concat([segment_pair[:, 0], segment_pair[:, 1]], axis=0)
    aux_pt_pair_indices = tf.concat([aux_join_pt0, aux_join_pt1], axis=0)
    e_weights = tf.ones(shape=(2 * n_seg_pairs))

    if segment_pair.shape[1] == 4:
        pt_indices = tf.concat(
            [pt_indices, segment_pair[:, 2], segment_pair[:, 3]], axis=0
        )
        aux_pt_pair_indices = tf.concat(
            [aux_pt_pair_indices, aux_join_pt0, aux_join_pt1], axis=0
        )
        e_weights = tf.concat(
            [e_weights, -tf.ones(shape=(2 * n_seg_pairs))], axis=0
        )

    seg_inds, pt_inds = original_graph.edges(etype="segment_has_point")
    aux_graph = dgl.heterograph(
        {
            ("point", "in", "segment"): (pt_inds, seg_inds),
            ("point", "join", "aux_pt_pair"): (
                pt_indices,
                aux_pt_pair_indices,
            ),
            ("aux_pt_pair", "join", "seg_pairs"): (
                tf.concat([aux_join_pt0, aux_join_pt1], axis=0),
                tf.concat([aux_join_pairs, aux_join_pairs], axis=0),
            ),
        }
    )  # .long()
    aux_graph.edges[("point", "join", "aux_pt_pair")].data[
        "weight"
    ] = e_weights
    return aux_graph


def _calculate_segment_pair_probabilities(aux_graph, point_logits):
    with aux_graph.local_scope():
        aux_graph.nodes["point"].data["logit"] = point_logits

        aux_graph.update_all(
            message_func=fn.u_mul_e("logit", "weight", "m"),
            reduce_func=fn.sum("m", "pt_pair_logit"),
            etype=("point", "join", "aux_pt_pair"),
        )
        aux_graph.nodes["aux_pt_pair"].data["pt_pair_logit"] = tf.math.tanh(
            aux_graph.nodes["aux_pt_pair"].data["pt_pair_logit"]
        )
        aux_graph.update_all(
            message_func=fn.copy_u("pt_pair_logit", "m"),
            reduce_func=fn.sum("m", "seg_pair_logit"),
            etype=("aux_pt_pair", "join", "seg_pairs"),
        )
        aux_graph.nodes["seg_pairs"].data["seg_pair_logit"] = -tf.math.abs(
            aux_graph.nodes["seg_pairs"].data["seg_pair_logit"]
        )
        probabilities = tf.transpose(
            dgl.softmax_nodes(aux_graph, "seg_pair_logit", ntype="seg_pairs")
        )
        return probabilities
