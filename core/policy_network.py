import dgl
from dgl.nn.tensorflow import HeteroGraphConv
from dgl.nn.tensorflow.conv import SAGEConv

import tensorflow as tf


class HeteroGraphPolicyNetwork(tf.keras.Model):
    def __init__(
        self,
        n_tri_feats=1,
        n_seg_feats=2,
        n_angle_feats=5,
        n_pt_feats=5,
        n_local_hidden_nodes_1=16,
        n_local_hidden_nodes_2=8,
        n_global_hidden_nodes_1=32,
        n_global_hidden_nodes_2=16,
    ):
        super().__init__()

        self._initalize_local_layers(
            n_tri_feats=n_tri_feats,
            n_seg_feats=n_seg_feats,
            n_angle_feats=n_angle_feats,
            n_pt_feats=n_pt_feats,
            n_local_hidden_nodes_1=n_local_hidden_nodes_1,
            n_local_hidden_nodes_2=n_local_hidden_nodes_2,
        )
        self._initialize_global_layers(
            n_global_hidden_nodes_1=n_global_hidden_nodes_1,
            n_global_hidden_nodes_2=n_global_hidden_nodes_2,
        )

    def call(self, triangulation):
        local_features = _prepare_local_features(triangulation)
        global_features = _prepare_global_features(triangulation)

        point_logits = self._call_local_layers(triangulation, local_features)
        triangulation_logits = self._call_global_layers(global_features)
        return point_logits, triangulation_logits

    def _initalize_local_layers(
        self,
        n_tri_feats=1,
        n_seg_feats=2,
        n_angle_feats=5,
        n_pt_feats=5,
        n_local_hidden_nodes_1=16,
        n_local_hidden_nodes_2=8,
    ):
        self.local_layer_1 = HeteroGraphConv(
            {
                "segment_in_triangle": SAGEConv(
                    (n_seg_feats, n_tri_feats),
                    n_local_hidden_nodes_1,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
                "segment_has_point": SAGEConv(
                    (n_seg_feats, n_pt_feats),
                    n_local_hidden_nodes_1,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
                "segment_bounds_angle": SAGEConv(
                    (n_seg_feats, n_angle_feats),
                    n_local_hidden_nodes_1,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
                "angle_at_point": SAGEConv(
                    (n_angle_feats, n_pt_feats),
                    n_local_hidden_nodes_1,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
                "triangle_contains_angle": SAGEConv(
                    (n_tri_feats, n_angle_feats),
                    n_local_hidden_nodes_1,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
            },
            aggregate="mean",
        )
        self.local_layer_2 = HeteroGraphConv(
            {
                "angle_at_point": SAGEConv(
                    (n_local_hidden_nodes_1, n_local_hidden_nodes_1),
                    n_local_hidden_nodes_2,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
                "triangle_contains_angle": SAGEConv(
                    (n_local_hidden_nodes_1, n_local_hidden_nodes_1),
                    n_local_hidden_nodes_2,
                    aggregator_type="mean",
                    activation=tf.math.tanh,
                ),
            },
            aggregate="mean",
        )
        self.local_layer_3 = HeteroGraphConv(
            {
                "angle_at_point": SAGEConv(
                    (n_local_hidden_nodes_2, n_local_hidden_nodes_2),
                    1,
                    aggregator_type="mean",
                    # activation=tf.math.tanh,
                ),
            },
            # aggregate="mean",
        )

    def _call_local_layers(self, graph, node_features):
        hidden = self.local_layer_1(graph, node_features)
        hidden = self.local_layer_2(graph, hidden)
        hidden = self.local_layer_3(graph, hidden)
        point_logits = hidden["point"]
        return point_logits

    def _initialize_global_layers(
        self,
        n_global_hidden_nodes_1=32,
        n_global_hidden_nodes_2=16,
    ):
        self.g_layer_1 = tf.keras.layers.Dense(
            n_global_hidden_nodes_1, activation=tf.math.tanh
        )
        self.g_layer_2 = tf.keras.layers.Dense(
            n_global_hidden_nodes_2, activation=tf.math.tanh
        )
        self.g_layer_3 = tf.keras.layers.Dense(7)

    def _call_global_layers(self, readout_features):
        hidden = self.g_layer_1(readout_features)
        hidden = self.g_layer_2(hidden)
        graph_logits = self.g_layer_3(hidden)
        return graph_logits


def _prepare_local_features(triangulation):
    tri_feats = tf.expand_dims(
        triangulation.nodes["triangle"].data["triangle_type"], 1
    )
    seg_feats = tf.concat(
        [
            tf.expand_dims(triangulation.nodes["segment"].data["boundary"], 1),
            tf.expand_dims(
                triangulation.nodes["segment"].data["segment_type"], 1
            ),
        ],
        axis=1,
    )
    angle_feats = tf.concat(
        [
            triangulation.nodes["angle"].data["angle_type"],
            tf.expand_dims(
                triangulation.nodes["angle"].data["light_cone_angle"], 1
            ),
        ],
        axis=1,
    )
    pt_feats = tf.concat(
        [
            triangulation.nodes["point"].data["n_angle_types"],
            tf.expand_dims(
                triangulation.nodes["point"].data["n_light_cone_angle"], 1
            ),
        ],
        axis=1,
    )

    local_features = {
        "triangle": tri_feats,
        "segment": seg_feats,
        "point": pt_feats,
        "angle": angle_feats,
    }
    return local_features


def _prepare_global_features(triangulation):
    n_triangles = triangulation.batch_num_nodes("triangle")
    n_segments = triangulation.batch_num_nodes("segment")
    n_points = triangulation.batch_num_nodes("point")

    # --------------------------------------------------------------------------
    log_n_tri = tf.math.log(
        tf.expand_dims(tf.cast(n_triangles, dtype=tf.float32), 1)
    )
    log_n_seg = tf.math.log(
        tf.expand_dims(tf.cast(n_segments, dtype=tf.float32), 1)
    )
    log_n_pt = tf.math.log(
        tf.expand_dims(tf.cast(n_points, dtype=tf.float32), 1)
    )
    # --------------------------------------------------------------------------
    frac_triangle_types = _mean_node_readout(
        n_triangles,
        _encode_types_for_node(
            triangulation.nodes["triangle"].data["triangle_type"]
        ),
    )

    # --------------------------------------------------------------------------
    encoded_segment_types = _encode_types_for_node(
        triangulation.nodes["segment"].data["segment_type"]
    )
    frac_segment_types = _mean_node_readout(n_segments, encoded_segment_types)

    boundary_segments = tf.expand_dims(
        triangulation.nodes["segment"].data["boundary"], 1
    )
    frac_boundary_segments = _mean_node_readout(n_segments, boundary_segments)

    frac_valid_segments = _mean_node_readout(
        n_segments, encoded_segment_types * boundary_segments
    )
    # --------------------------------------------------------------------------
    mean_complete_light_cones = _mean_node_readout(
        n_points,
        tf.expand_dims(
            triangulation.nodes["point"].data["n_light_cone_angle"], 1
        )
        / 4,
    )
    mean_angle_types = _mean_node_readout(
        n_points, triangulation.nodes["point"].data["n_angle_types"]
    )

    global_features = tf.concat(
        [
            log_n_tri,
            log_n_seg,
            log_n_pt,
            frac_triangle_types,
            frac_segment_types,
            frac_boundary_segments,
            frac_valid_segments,
            mean_complete_light_cones,
            mean_angle_types,
        ],
        axis=1,
    )
    return global_features


def _encode_types_for_node(types_for_node):
    encoded_types = tf.one_hot(tf.cast(types_for_node, dtype=tf.int32), 2)
    return encoded_types


def _mean_node_readout(n_nodes, data):
    readout = dgl.ops.segment.segment_reduce(n_nodes, data, reducer="mean")
    return readout
