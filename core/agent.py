import tensorflow as tf


def extract_endpoint_pair_combinations(
    segment_to_point_adj: tf.Tensor,
    valid_segments: tf.Tensor,
    n_light_crossings_per_pt: tf.Tensor,
) -> (tf.Tensor, tf.Tensor):
    endpt_adj_of_vseg = _construct_endpoint_adjacency_of_valid_segments(
        segment_to_point_adj, valid_segments
    )
    valid_point_combinations_filter = _consolidate_point_combination_filters(
        segment_to_point_adj,
        endpt_adj_of_vseg,
        n_light_crossings_per_pt,
    )

    segment_endpts = _get_valid_segment_endpoints_to_connect(
        endpt_adj_of_vseg,
    )
    valid_endpt_pair_combinations_filter = (
        _create_combination_filter_for_each_endpoint_pair(
            segment_endpts,
            valid_point_combinations_filter,
            endpt_adj_of_vseg,
        )
    )
    (
        current_pair_combos,
        new_tri_pair_combos,
    ) = _get_compatible_endpoint_pair_combinations(
        segment_endpts, valid_endpt_pair_combinations_filter
    )
    return current_pair_combos, new_tri_pair_combos


def _get_compatible_endpoint_pair_combinations(
    segment_endpts: tf.Tensor, valid_endpt_pair_combinations_filter: tf.Tensor
) -> (tf.Tensor, tf.Tensor):
    """
    Gathers the node indices of compatible endpoint pairs. Along with the
    segment of the endpoint pairs, these can then be merged downstream. That is,
    [s = (p0, p1), s' = (p0', p1')] -> S = (P0, P1), where the first points and
    the second points are merged, respectively: [p0, p0'] -> P0 and
    [p1, p1'] -> P1.

    If a new triangle is added, the endpoint pairs (p0, p1) that are compatible
    with the six new endpoint pair types (q0, q1) are also gathered. The new
    endpoint pair types refer to:
        0: (A, A') -> Time-like segment of a tss triangle
        1: (A, B) -> Time-like segment of a stt triangle
        2: (B, A) -> Time-like segment of a stt triangle (opposite orientation)
        3: (A, C) -> Space-like segment of a tss triangle
        4: (C, A) -> Space-like segment of a tss triangle (opposite orientation)
        5: (A, A') -> Space-like segment of a stt triangle
        where the point types are:
        A -> Point incident to one time-like segment and one space-like segment
        B -> Point incident to two time-like segments
        C -> Point incident to two space-like segments

    Parameters
    ----------
    segment_endpts: tf.Tensor
        Tensor of shape (N+6, 3) corresponding to ordered endpoints of boundary
        segments for each segment type. N is the current number of ordered valid
        segment endpoints.
            -> segment_endpts[:, 0] corresponds to the segment type
            -> segment_endpts[:, 1:] are the node indices.
    valid_endpt_pair_combinations_filter: tf.Tensor
        Tensor of shape (N+6, n_points, n_points) representing the endpoint
        pairs (p0', p1') compatible with the endpoint pairs (p0, p1) in
        segment_endpts

    Returns
    -------
    Tuple:
        current_pair_combos: tf.Tensor
            Tensor of shape (M, 5) representing the node indices of the endpoint
            pairs being matched. M is the number of matched endpoint pairs that
            can be glued downstream [(p0, p1), (p0', p1')] -> (P0, P1)
                -> current_pair_combos[:, 0] refer to the segment type of the
                    segment where the endpoint pairs belong
                -> current_pair_combos[:, [1, 2]] refer to the first pairs of
                    endpoints to be matched (p0, p1)
                -> current_pair_combos[:, [3, 4]] refer to the second pairs of
                    endpoints to be matched (p0', p1')
        new_tri_pair_combos: tf.Tensor
            Tensor of shape (M', 3) representing the node indices of endpoint
            pairs (p0, p1) that can be matched and glued with possibly new pairs
            from new triangles (q0, q1). M' is the number of such endpoint pairs
                -> new_tri_pair_combos[:, 0] refer to the endpoint pair types
                    of the new triangles
                -> new_tri_pair_combos[:, [1, 2]] refer to the endpoint pairs
                    that can be matched to the new endpoint pair types
    """
    valid_combos_for_connected_endpt_pairs = tf.where(
        valid_endpt_pair_combinations_filter[:-6, :, :]
    )
    endpoint_pairs_being_considered = tf.gather(
        segment_endpts[:-6, :],
        valid_combos_for_connected_endpt_pairs[:, 0],
    )
    current_pair_combos = tf.concat(
        [
            endpoint_pairs_being_considered,
            valid_combos_for_connected_endpt_pairs[:, 1:],
        ],
        axis=1,
    )

    new_tri_pair_combos = tf.where(
        valid_endpt_pair_combinations_filter[-6:, :, :]
    )
    return current_pair_combos, new_tri_pair_combos


def _create_combination_filter_for_each_endpoint_pair(
    segment_endpts: tf.Tensor,
    valid_point_combinations_filter: tf.Tensor,
    endpt_adj_of_vseg: tf.Tensor,
) -> tf.Tensor:
    """
    For each endpoint pair (p0, p1) in segment_endpts, generate compatible
    endpoint pairs (p0', p1') as determined by valid_point_combinations_filter
    and endpt_adj_of_vseg. That is, p0' is a point compatible with p0, and
    similarly with p1' being a point compatible with p1. Additionally, the pair
    (p0', p1') should be enpoints of a valid segment.

    Downstream, the segments defined by the matched endpoint pairs
    s = (p0, p1) and s' = (p0', p1') can be glued together [s, s'] -> S with the
    orientation of the gluing being defined by how the endpoints are glued
    [p0, p0'] -> P0 and [p1, p1'] -> P1.

    Parameters
    ----------
    segment_endpts: tf.Tensor
        Tensor of shape (N+6, 3) corresponding to ordered endpoints of boundary
        segments for each segment type. N is the current number of ordered valid
        segment endpoints.
            -> segment_endpts[:, 0] corresponds to the segment type
            -> segment_endpts[:, 1:] are the node indices.
    valid_point_combinations_filter: tf.Tensor
        Boolean tensor of shape (n_segment_types, n_points+3, n_points)
        representing whether the point pairs can be glued together
    endpt_adj_of_vseg: tf.Tensor
        Tensor of shape (n_segment_types, n_points, n_points) representing the
        points connected through a boundary segment of a particular type

    Returns
    -------
    valid_endpt_pair_combinations_filter: tf.Tensor
        Tensor of shape (N+6, n_points, n_points) representing the endpoint
        pairs (p0', p1') compatible with the endpoint pairs (p0, p1) in
        segment_endpts
    """
    # ------------ Get the compatible points for p0 and p1 separately ----------
    lower_tri_pt_combo_filter = tf.linalg.band_part(
        valid_point_combinations_filter, -1, 0
    )
    endpt_0_combos = _get_combination_filter_for_each_point(
        segment_endpts[:, :2], lower_tri_pt_combo_filter
    )
    endpt_1_combos = _get_combination_filter_for_each_point(
        segment_endpts[:, ::2], lower_tri_pt_combo_filter
    )

    # --- Create filter for points pairs (p0', p1') compatible with (p0, p1) ---
    endpt_0s_filter = tf.expand_dims(endpt_0_combos, 2)
    endpt_1s_filter = tf.expand_dims(endpt_1_combos, 1)
    endpt_pair_combos_filter = tf.cast(
        tf.math.logical_and(endpt_0s_filter, endpt_1s_filter), tf.float32
    )

    # - Get only the pairs (p0', p1') that are endpoints of a valid segment s' -
    lower_tri_neighbors = tf.linalg.band_part(endpt_adj_of_vseg, -1, 0)
    valid_pt_neighbors_repeat = tf.gather(
        lower_tri_neighbors, segment_endpts[:, 0]
    )
    valid_endpt_pair_combinations_filter = (
        valid_pt_neighbors_repeat * endpt_pair_combos_filter
    )
    return valid_endpt_pair_combinations_filter


def _get_combination_filter_for_each_point(
    points: tf.Tensor, point_combinations_filter: tf.Tensor
) -> tf.Tensor:
    """
    Get the possible points that can be paired with the input points being
    considered. These possible points are represented as a filter.

    Parameters
    ----------
    points: tf.Tensor
        Tensor of shape (N, 2) corresponding to the points being considered.
            -> points[:, 0] refers to the segment type where the points belong
            -> points[:, 1] refers to the node indices of the points.
    point_combinations_filter: tf.Tensor
        Boolean tensor of shape (n_segment_types, n_points+3, n_points)
        representing whether the point pairs can be glued together

    Returns
    -------
    combination_filter_for_each_pt: tf.Tensor
        Boolean tensor of shape (N, n_points) representing the rows of
        point_combinations_filter for the indices in points.
    """
    combination_filter_for_each_pt = tf.gather_nd(
        point_combinations_filter,
        points,
    )
    return combination_filter_for_each_pt


def _get_valid_segment_endpoints_to_connect(
    endpt_adj_of_vseg: tf.Tensor,
) -> tf.Tensor:
    """
    Generates the node indices of boundary segment endpoint pairs (p0, p1) for
    each segment type.

    Downstream, each endpoint p is matched with a compatible point p'
    as specified by the point combination filters. So, the valid segment
    endpoint pairs (p0, p1) will be matched with compatible points (p0', p1').
    Additionally, the compatible points (p0', p1') must themselves be endpoints
    of a particular valid segment. Thus, a valid segment will be matched with
    another valid segment. The segments can then be glued together and the
    specific point pair matching determines how the segments were oriented when
    glued together.

    To account for new triangles being added, their segment endpoints are also
    considered. Note that these enpoints are distinct even if their types can
    be the same. For each segment type, the enpoints are:
         Time-like segments:
            (A, A') -> Segment with the same endpoint type
                    -> For a tss triangle
            (A, B) and (B, A) -> Segment with A and B endpoint types
                    -> For a stt triangle
        Space-like segments:
            (A, C) and (C, A) -> Segment with A and C endpoint types
                    -> For a tss triangle
            (A, A') -> Segment with the same endpoint type
                    -> For a stt triangle
        where the point types are:
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
    segment_endpts: tf.Tensor
        Tensor of shape (N+6, 3) corresponding to ordered endpoints of boundary
        segments for each segment type. N is the current number of ordered valid
        segment endpoints.
            -> segment_endpts[:, 0] corresponds to the segment type
            -> segment_endpts[:, 1:] are the node indices.
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


def _consolidate_point_combination_filters(
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
            -> valid_segments[0, :, :] represent time-like boundary segments
            -> valid_segments[1, :, :] represent space-like boundary segments

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
