from copy import deepcopy

import dgl
import dgl.function as fn
import tensorflow as tf


class TriangulationEnvironment:
    def __init__(self):
        triangle = _create_base_triangle()
        self.tss_triangle = _create_tss_triangle(triangle)
        self.stt_triangle = _create_stt_triangle(triangle)

        self.state = None

    def reset(self):
        random_start = tf.random.uniform(shape=(), minval=0, maxval=1)
        self.state = tf.case(
            [(tf.math.less(random_start, 0.5), lambda: self.tss_triangle)],
            default=lambda: self.stt_triangle,
        )
        return self.state


def _create_base_triangle():
    triangle = dgl.heterograph(
        {
            ("segment", "segment_in_triangle", "triangle"): (
                tf.constant([0, 1, 2]),
                tf.constant([0, 0, 0]),
            ),
            ("segment", "segment_has_point", "point"): (
                tf.constant([0, 0, 1, 1, 2, 2]),
                tf.constant([0, 1, 1, 2, 2, 0]),
            ),
            ("segment", "segment_bounds_angle", "angle"): (
                tf.constant([0, 2, 0, 1, 1, 2]),
                tf.constant([0, 0, 1, 1, 2, 2]),
            ),
            ("angle", "angle_at_point", "point"): (
                tf.constant([0, 1, 2]),
                tf.constant([0, 1, 2]),
            ),
            ("triangle", "triangle_contains_angle", "angle"): (
                tf.constant([0, 0, 0]),
                tf.constant([0, 1, 2]),
            ),
        }
    )
    return triangle


def _create_tss_triangle(triangle):
    tss_triangle = deepcopy(triangle)
    tss_triangle.nodes["segment"].data["segment_type"] = tf.constant([0, 1, 1])
    tss_triangle = _create_triangle_data(tss_triangle)
    tss_triangle = _update_triangulation_data(tss_triangle)
    return tss_triangle


def _create_stt_triangle(triangle):
    stt_triangle = deepcopy(triangle)
    stt_triangle.nodes["segment"].data["segment_type"] = tf.constant([1, 0, 0])
    stt_triangle = _create_triangle_data(stt_triangle)
    stt_triangle = _update_triangulation_data(stt_triangle)
    return stt_triangle


def _check_light_cone_angle(node):
    segments_type = tf.cast(tf.transpose(node.mailbox["m"]), dtype=tf.bool)
    light_cone_angle = tf.cast(
        tf.math.logical_xor(*segments_type), dtype=tf.float32
    )
    return {"light_cone_angle": light_cone_angle}


def _check_triangle_type(node):
    triangle_type = tf.reduce_sum(node.mailbox["m"], axis=-1) - 1
    return {"triangle_type": triangle_type}


def _calculate_encoded_angle(node):
    angle_type = (
        2 * tf.reshape(node.mailbox["m"], shape=(-1,))
        + node.data["light_cone_angle"]
    )
    encoded_angle_type = tf.one_hot(tf.cast(angle_type, tf.int32), 4)
    return {"angle_type": encoded_angle_type}


def _create_triangle_data(graph):
    graph.update_all(
        message_func=fn.copy_u("segment_type", "m"),
        reduce_func=_check_light_cone_angle,
        etype="segment_bounds_angle",
    )

    graph.update_all(
        message_func=fn.copy_u("segment_type", "m"),
        reduce_func=_check_triangle_type,
        etype="segment_in_triangle",
    )

    graph.update_all(
        message_func=fn.copy_u("triangle_type", "m"),
        reduce_func=_calculate_encoded_angle,
        etype="triangle_contains_angle",
    )
    return graph


def _update_triangulation_data(triangulation):
    triangulation.nodes["segment"].data["boundary"] = tf.cast(
        tf.math.equal(
            triangulation.out_degrees(etype="segment_in_triangle"), 1
        ),
        dtype=tf.float32,
    )
    triangulation.update_all(
        message_func=fn.copy_u("light_cone_angle", "m"),
        reduce_func=fn.sum("m", "n_light_cone_angle"),
        etype="angle_at_point",
    )
    triangulation.update_all(
        message_func=fn.copy_u("angle_type", "m"),
        reduce_func=fn.sum("m", "n_angle_types"),
        etype="angle_at_point",
    )
    return triangulation