# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:06:23 2020

@author: sujeet.kumar
"""


from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import matplotlib.pyplot as plt

# create the nodes
ServiceType = BbnNode(Variable(0, 'ServiceType', ['A', 'B','C']), [0.48,
0.251,
0.269
])
ClaimCost = BbnNode(Variable(1, 'ClaimCost', ['lt 200','501-1000','gt 1000']), [0.40,	0.56,	0.04,
0.75,	0.18,	0.07,
0.64,	0.31,	0.05

 ])

PastCall = BbnNode(Variable(2, 'PastCall', ['0-4','_4+']), [0.829,  0.171


])
Detractors = BbnNode(Variable(3, 'Detractors', ['Y', 'N']), [0.10,	0.90,
0.67,	0.33,
0.12,	0.88,
0.60,	0.40,
0.06,	0.94,
0.00,	0.00,
0.10,	0.90,
0.70,	0.30,
0.14,	0.86,
0.50,	0.50,
0.06,	0.94,
0.00,	0.00,
0.30,	0.70,
0.79,	0.21,
0.57,	0.43,
0.74,	0.26,
0.33,	0.67,
1.00,	0.00

])




# create the network structure
bbn = Bbn() \
    .add_node(ServiceType) \
    .add_node(ClaimCost) \
    .add_node(PastCall) \
    .add_node(Detractors) \
    .add_edge(Edge(ServiceType, ClaimCost, EdgeType.DIRECTED)) \
    .add_edge(Edge(ServiceType, Detractors, EdgeType.DIRECTED)) \
    .add_edge(Edge(ClaimCost, Detractors, EdgeType.DIRECTED)) \
    .add_edge(Edge(PastCall, Detractors, EdgeType.DIRECTED))



# %matplotlib inline
# from pybbn.generator.bbngenerator import convert_for_drawing
# import matplotlib.pyplot as plt
# import networkx as nx
# import warnings


# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
    
#     graph = convert_for_drawing(bbn)
#     pos = nx.nx_agraph.graphviz_layout(graph,prog = 'neato') 
#     '''(graph, prog='neato')'''
#     plt.figure(figsize=(20, 10))
#     plt.subplot(121)
#     labels = dict([(k, node.variable.name) for k, node in bbn.nodes.items()])
#     nx.draw(graph, pos=pos, with_labels=True, labels=labels)
#     plt.title('BBN DAG')




# convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# insert an observation evidence
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('PastCall')) \
    .with_evidence('0-4', 1) \
    .build()
join_tree.set_observation(ev)

# print the marginal probabilities
for node in join_tree.get_bbn_nodes():
    potential = join_tree.get_bbn_potential(node)
    print(node)
    print(potential)






