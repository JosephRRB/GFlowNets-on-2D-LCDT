import tensorflow as tf


# class PolicyNetwork:
#     def __init__(self, ):


def _get_valid_segment_endpoints_to_connect(
    endpt_adj_of_vseg: tf.Tensor,
) -> tf.Tensor:
    """


    :param endpt_adj_of_vseg:
    :return:
    """
    original_segment_endpts = tf.where(endpt_adj_of_vseg)
    n_original_pts = endpt_adj_of_vseg.shape[1]
    endpts_of_new_triangle = tf.constant(
        [
            [0, n_original_pts, n_original_pts],
            [0, n_original_pts, n_original_pts + 1],
            [0, n_original_pts + 1, n_original_pts],
            [1, n_original_pts, n_original_pts + 2],
            [1, n_original_pts + 2, n_original_pts],
            [1, n_original_pts, n_original_pts],
        ],
        dtype=tf.int64,
    )
    segment_endpts = tf.concat(
        [original_segment_endpts, endpts_of_new_triangle], axis=0
    )
    return segment_endpts


def _combine_possible_point_combination_filters(
    segment_to_point_adj: tf.Tensor,
    endpt_adj_of_vseg: tf.Tensor,
    n_light_crossings_per_pt: tf.Tensor,
) -> tf.Tensor:
    """
    Filters out point combinations that:
        - are nearest neighbors (connected by a segment)
        - don't have boundary segments of the same type
        - violate local causality

    Downstream, the point pairs that are left can then be chosen to be glued
    with each other.

    Parameters
    ----------
    segment_to_point_adj: tf.Tensor
        Tensor of shape (n_segments, n_points) representing which segments
        contain which points
    endpt_adj_of_vseg: tf.Tensor
        Tensor of shape (n_segment_types, n_points, n_points) representing the
        points connected through a boundary segment of a particular type
    n_light_crossings_per_pt: tf.Tensor
        Tensor of shape (n_points, ) representing the number of light cone
        crossings around each point

    Returns
    -------
    valid_point_combinations_filter: tf.Tensor
        Boolean tensor of shape (n_segment_types, n_points+3, n_points)
        representing whether the point pairs can be glued together
    """
    non_neighbor_points_filter = (
        _create_filter_for_non_neighbor_point_combinations(
            segment_to_point_adj
        )
    )
    endpoints_of_valid_segments_filter = (
        _create_filter_for_combinations_of_valid_segment_endpoints(
            endpt_adj_of_vseg
        )
    )
    points_with_light_cones_filter = (
        _create_filter_for_point_combinations_obeying_local_causality(
            n_light_crossings_per_pt
        )
    )
    valid_point_combinations_filter = tf.math.logical_and(
        tf.math.logical_and(
            endpoints_of_valid_segments_filter, points_with_light_cones_filter
        ),
        non_neighbor_points_filter,
    )
    return valid_point_combinations_filter


def _create_filter_for_non_neighbor_point_combinations(
    segment_to_point_adj: tf.Tensor,
) -> tf.Tensor:
    """
    Filters out point pair combinations that are connected by a segment. What
    remains are then non-neighboring point pairs. Downstream, these point pairs
    can be glued into one point. This filter then removes the possibility of
    gluing the two endpoints of a segment into one.

    To account for new triangles being added, the possibility of the original
    points to be paired with the three new point types is also considered. In
    this case, the new point types of the new triangle will always be
    disconnected from the original points which always make them non-neighbors.

    Parameters
    ----------
    segment_to_point_adj: tf.Tensor
        Tensor of shape (n_segments, n_points) representing which segments
        contain which points

    Returns
    -------
    non_neighbor_points_filter: tf.Tensor
        Boolean tensor of shape (n_points+3, n_points) representing whether
        the point pairs are not connected by a segment
    """
    pt_to_pt = _construct_pt_to_pt_adjacency(segment_to_point_adj)
    # non-neighbor points are points that are not connected by a segment
    non_neighbor_points_filter = tf.math.equal(pt_to_pt, 0)

    # points for new triangle
    # 3 point types for tss or stt triangle
    # new points are always initially disconnected
    disconnected_new_points = tf.cast(
        tf.ones(shape=(3, segment_to_point_adj.shape[1])), tf.bool
    )
    non_neighbor_points_filter = tf.concat(
        [non_neighbor_points_filter, disconnected_new_points], axis=0
    )
    return non_neighbor_points_filter


def _create_filter_for_combinations_of_valid_segment_endpoints(
    endpt_adj_of_vseg: tf.Tensor,
) -> tf.Tensor:
    """
    Filters out point pair combinations that are not endpoints of a particularly
    typed boundary segment. Two points can only be paired if they are contained
    by boundary segments of the same type. Downstream, these point pairs can be
    glued into one point along with the valid segments they are contained in.
    Thus, two distinct points can be paired if each has at least one valid
    segment while a point can be paired with itself if it has at least two valid
    segments.

    To account for new triangles being added, the possibility of the original
    points to be paired with the three new point types is also considered.
    Point types of new triangles:
         A -> Point incident to one time-like segment and one space-like segment
         B -> Point incident to two time-like segments
         C -> Point incident to two space-like segments

    Parameters
    ----------
    endpt_adj_of_vseg: tf.Tensor
        Tensor of shape (n_segment_types, n_points, n_points) representing the
        points connected through a boundary segment of a particular type

    Returns
    -------
    vseg_endpts_filter: tf.Tensor
        Boolean tensor of shape (n_segment_types, n_points+3, n_points)
        representing whether the point pairs are endpoints of valid segments
    """
    n_vseg_per_pt = tf.reduce_sum(endpt_adj_of_vseg, axis=1)
    # combinations of different points should have at least one valid segment
    pts_with_vseg = tf.math.greater_equal(n_vseg_per_pt, 1)

    # points for new triangle
    # 3 point types for tss or stt triangle
    # points having at least one valid segment
    new_pts_with_vseg = tf.constant(
        [[[True], [True], [False]], [[True], [False], [True]]]
    )

    pts_to_consider = tf.concat(
        [tf.expand_dims(pts_with_vseg, 2), new_pts_with_vseg], axis=1
    )

    endpts_combos = tf.math.logical_and(
        tf.expand_dims(pts_with_vseg, 1), pts_to_consider
    )
    # same points should have (at least?) two valid segments
    endpts_common = tf.math.greater_equal(pts_with_vseg, 2)
    vseg_endpts_filter = tf.linalg.set_diag(endpts_combos, endpts_common)
    return vseg_endpts_filter


def _create_filter_for_point_combinations_obeying_local_causality(
    n_light_crossings_per_pt: tf.Tensor,
) -> tf.Tensor:
    """
    Filters out point pair combinations that violate local causality. Local
    causality is obeyed when an internal point has exactly four light cone
    crossings. Whereas points on the boundary segments should have four or less.
    The number of light cone crossings around a point corresponds to the number
    of times the type of the incident segments change. For example, there is a
    light cone crossing between a time-like and a space-like segment.
    Downstream, the point pairs can be glued into one point. So, distinct points
    that are paired (on the boundary) should have, at most, four light cone
    crossings in total. If a point is paired with itself, this should yield an
    internal point after its incident boundary segments are glued together. A
    point paired with itself should then have exactly four light cone crossings.

    To account for new triangles being added, the possibility of the original
    points to be paired with the three new point types is also considered.
    Point types of new triangles:
         A -> Has one light cone crossing
         B -> Has none
         C -> Has none

    Parameters
    ----------
    n_light_crossings_per_pt: tf.Tensor
        Tensor of shape (n_points, ) representing the number of light cone
        crossings around each point

    Returns
    -------
    light_cones_filter: tf.Tensor
        Boolean tensor of shape (n_points+3, n_points) representing whether the
        point pairs will obey local causality when glued together (resulting
        point should have at most four light cone crossings for boundary points,
        exactly four for internal points)
    """
    # points for new triangle
    # 3 point types for tss or stt triangle
    # number of light cone crossings for each point type
    new_pts_n_light_crossings = tf.constant([[1], [0], [0]], dtype=tf.float32)

    pts_to_consider = tf.concat(
        [
            tf.expand_dims(n_light_crossings_per_pt, 1),
            new_pts_n_light_crossings,
        ],
        axis=0,
    )

    n_light_crossings_per_pt_combos = (
        tf.expand_dims(n_light_crossings_per_pt, 0) + pts_to_consider
    )
    # combinations of different points (on the boundary) should have at most
    # four light cone crossings
    off_diag_light_cones = tf.math.less_equal(
        n_light_crossings_per_pt_combos, 4
    )
    # same points should yield internal points which should have exactly four
    # light cone crossings
    diag_light_cones = tf.math.equal(n_light_crossings_per_pt, 4)
    light_cones_filter = tf.linalg.set_diag(
        off_diag_light_cones, diag_light_cones
    )
    return light_cones_filter


def _construct_endpoint_adjacency_of_valid_segments(
    segment_to_point_adj: tf.Tensor, valid_segments: tf.Tensor
) -> tf.Tensor:
    """
    Constructs the adjacency matrix of points connected by boundary segments
    of a particular type (which we call valid segments).

    Parameters
    ----------
    segment_to_point_adj: tf.Tensor
        Tensor of shape (n_segments, n_points) representing which segments
        contain which points
    valid_segments: tf.Tensor
        Tensor of shape (n_segment_types, n_segments, 1) representing whether
        the segment is a boundary segment and of a particular type
        - Boundary segments: Segments that are contained in only one triangle
        - Segment types: Time-like type or Space-like type
        - valid_segments[0, :, :] represent time-like boundary segments while
            valid_segments[1, :, :] represent space-like boundary segments

    Returns
    -------
    endpt_adj_of_vseg: tf.Tensor
        Tensor of shape (n_segment_types, n_points, n_points) representing the
        points connected through a boundary segment of a particular type
    """
    vseg_to_pt = valid_segments * segment_to_point_adj
    endpt_adj_of_vseg = _construct_pt_to_pt_adjacency(vseg_to_pt)
    return endpt_adj_of_vseg


def _construct_pt_to_pt_adjacency(
    segment_to_point_adj: tf.Tensor,
) -> tf.Tensor:
    """
    Constructs the adjacency matrix of points connected by segments

    Parameters
    ----------
    segment_to_point_adj: tf.Tensor
        Tensor of shape (n_segments, n_points) representing which segments
        contain which points

    Returns
    -------
    pt_to_pt_adj: tf.Tensor
        Tensor of shape (n_points, n_points) representing the points connected
        through a segment
    """
    pt_degrees = tf.reduce_sum(segment_to_point_adj, axis=-2)
    pt_to_pt_adj = tf.linalg.matmul(
        segment_to_point_adj, segment_to_point_adj, transpose_a=True
    ) - tf.linalg.diag(pt_degrees)
    return pt_to_pt_adj
