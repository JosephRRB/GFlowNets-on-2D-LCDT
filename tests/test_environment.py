import tensorflow as tf

from core.environment import TriangulationEnvironment


def test_reset_gives_one_complete_triangle():
    environment = TriangulationEnvironment()
    state = environment.reset()

    seg_to_tri_adj = state.adj(etype="segment_in_triangle")
    seg_to_pt_adj = state.adj(etype="segment_has_point")
    seg_to_angle_adj = state.adj(etype="segment_bounds_angle")
    angle_to_pt_adj = state.adj(etype="angle_at_point")
    tri_to_angle_adj = state.adj(etype="triangle_contains_angle")

    dense_tri_to_seg = tf.sparse.to_dense(tf.sparse.reorder(seg_to_tri_adj))
    dense_seg_to_pt = tf.sparse.to_dense(tf.sparse.reorder(seg_to_pt_adj))
    dense_seg_to_angle = tf.sparse.to_dense(tf.sparse.reorder(seg_to_angle_adj))
    dense_angle_to_pt = tf.sparse.to_dense(tf.sparse.reorder(angle_to_pt_adj))
    dense_tri_to_angle = tf.sparse.to_dense(tf.sparse.reorder(tri_to_angle_adj))

    expected_tri_to_seg_adj = tf.constant([[1, 1, 1]], dtype=tf.float32)
    expected_seg_to_pt_adj = tf.constant(
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=tf.float32
    )
    expected_seg_to_angle_adj = tf.constant(
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=tf.float32
    )
    expected_angle_to_pt_adj = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32
    )
    expected_tri_to_angle_adj = tf.constant([[1, 1, 1]], dtype=tf.float32)

    tf.assert_equal(dense_tri_to_seg, expected_tri_to_seg_adj)
    tf.assert_equal(dense_seg_to_pt, expected_seg_to_pt_adj)
    tf.assert_equal(dense_seg_to_angle, expected_seg_to_angle_adj)
    tf.assert_equal(dense_angle_to_pt, expected_angle_to_pt_adj)
    tf.assert_equal(dense_tri_to_angle, expected_tri_to_angle_adj)


# def test_reset_gives_correct_tss_triangle_data():
#     tf.random.set_seed(1337)
#     environment = TriangulationEnvironment()
#     state = environment.reset()
#
#     segment_data = state.nodes["segment"].data["type"]
#     point_data = state.nodes["point"].data["angle"]
#
#     expected_segment_data = tf.constant([0, 1, 1])
#     expected_point_data = tf.constant(
#         [
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [0, 0, 1, 0, 0],
#         ]
#     )
#
#     tf.assert_equal(segment_data, expected_segment_data)
#     tf.assert_equal(point_data, expected_point_data)
#
#
# def test_reset_gives_correct_stt_triangle_data():
#     tf.random.set_seed(42)
#     environment = TriangulationEnvironment()
#     state = environment.reset()
#
#     segment_data = state.nodes["segment"].data["type"]
#     point_data = state.nodes["point"].data["angle"]
#
#     expected_segment_data = tf.constant([1, 0, 0])
#     expected_point_data = tf.constant(
#         [
#             [1, 0, 0, 1, 0],
#             [1, 0, 0, 1, 0],
#             [0, 0, 0, 0, 1],
#         ]
#     )
#
#     tf.assert_equal(segment_data, expected_segment_data)
#     tf.assert_equal(point_data, expected_point_data)
