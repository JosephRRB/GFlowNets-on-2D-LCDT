import dgl

from core.agent.endpoint_pair_combinations import extract_endpoint_pair_combinations


def test():
    ### TODO: check what happens if no timelike segments, what happens to endpt_adj_of_vseg
    triangulation = dgl.load_graphs("./data/test_triangulation")[0][0]

    combinations = extract_endpoint_pair_combinations(triangulation)

    # time_like_pairs, space_like_pairs = current
    # time_like_pairs_for_tss_triangle, space_like_pairs_for_tss_triangle = tss
    # time_like_pairs_for_stt_triangle, space_like_pairs_for_stt_triangle = stt
    #
    # # tf.concat(tss, axis=0).shape
    # current


def test0():
    from core.agent.policy_network import HeteroGraphPolicyNetwork

    # from dgl.nn.tensorflow import GraphConv

    triangulation = dgl.load_graphs("./data/test_triangulation")[0][0]
    # triangulation = dgl.batch([triangulation] * 3)

    policy = HeteroGraphPolicyNetwork()
    point_logits, triangulation_logits = policy(triangulation)
    point_logits


# def test1():
#     from core.agent import Agent
#
#     triangulation = dgl.load_graphs("./data/test_triangulation")[0][0]
#     agent = Agent()
#     agent.act(triangulation)


def test2():
    triangulation = dgl.load_graphs("./data/test_triangulation_1")[0][0]
    endpt_combos = extract_endpoint_pair_combinations(triangulation)
    endpt_combos
