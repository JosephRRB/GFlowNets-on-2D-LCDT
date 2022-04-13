from copy import deepcopy

import dgl
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
            default=lambda: self.stt_triangle
        )
        return self.state


def _create_base_triangle():
    triangle = dgl.heterograph(
        {
            ("triangle", "has", "segment"): (
                tf.constant([0, 0, 0]),
                tf.constant([0, 1, 2]),
            ),
            ("segment", "has", "point"): (
                tf.constant([0, 0, 1, 1, 2, 2]),
                tf.constant([0, 1, 1, 2, 2, 0]),
            ),
        }
    )
    return triangle


def _create_tss_triangle(triangle):
    tss_triangle = deepcopy(triangle)
    tss_triangle.nodes["segment"].data["type"] = tf.constant([0, 1, 1])
    tss_triangle.nodes["point"].data["angle"] = tf.constant(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    return tss_triangle


def _create_stt_triangle(triangle):
    stt_triangle = deepcopy(triangle)
    stt_triangle.nodes["segment"].data["type"] = tf.constant([1, 0, 0])
    stt_triangle.nodes["point"].data["angle"] = tf.constant(
        [
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    return stt_triangle
