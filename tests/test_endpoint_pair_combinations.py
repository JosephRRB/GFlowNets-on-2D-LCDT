import dgl
import tensorflow as tf

from core.endpoint_pair_combinations import extract_endpoint_pair_combinations


def test():
    ### TODO: check what happens if no timelike segments, what happens to endpt_adj_of_vseg
    triangulation = dgl.load_graphs("./data/test_triangulation")[0][0]

    current, tss, stt = extract_endpoint_pair_combinations(triangulation)
    time_like_pairs, space_like_pairs = current
    time_like_pairs_for_tss_triangle, space_like_pairs_for_tss_triangle = tss
    time_like_pairs_for_stt_triangle, space_like_pairs_for_stt_triangle = stt

    # tf.concat(tss, axis=0).shape
    current

def test2():
    triangulation0 = dgl.load_graphs("./data/test_triangulation")[0][0]

    from copy import deepcopy
    triangulation1 = deepcopy(triangulation0)
    triangulation1.nodes['segment'].data['type'] = (
        triangulation1.nodes['segment'].data['type'] + 1) % 2

    batch_triangulation = dgl.batch([triangulation0, triangulation1])
    # dgl.unbatch(batch_triangulation)
    current, tss, stt = extract_endpoint_pair_combinations(batch_triangulation)


