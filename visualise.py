import graphviz
import cv2


def visualize_neural_network(
        connectivities,
        skip_connectivity=None):
    dot = graphviz.Digraph()

    neurons_in_layers = [set()]

    for i in range(len(connectivities)):
        neurons_in_layers.append(set())
        for to, from_ in connectivities[i]:
            neurons_in_layers[i].add(from_)
            neurons_in_layers[i+1].add(to)

    # Add nodes
    # for ith_layer, n_neurons in enumerate(neurons_in_layers):
    #     for jth_neuron in n_neurons:
    #         dot.node(str(ith_layer) + str(jth_neuron), str(jth_neuron))

    for ith_layer, connectivities in enumerate(connectivities):
        for (to, from_) in connectivities[:10]:
            dot.edge(str(ith_layer) + str(from_), str(ith_layer+1) + str(to))
    # # for i, layer in enumerate(layers):
    # #     dot.node(str(i), str(layer))

    # # Add edges
    # for i in range(len(layers) - 1):
    #     dot.edge(str(i), str(i + 1))

    # Render the graph
    dot.render('neural_network.gv', format='png')

    # Read the image using OpenCV
    image = cv2.imread('neural_network.gv.png')

    # Display the image using OpenCV
    cv2.imshow('Neural Network', image)
    cv2.waitKey(100)


connectivities = ([[2, 17],
  [2, 21],
  [2, 25],
  [5, 33]], [[0, 2],
             [0, 5],
             [1, 2]])

# # Example usage
# # Example network with three layers of sizes 10, 20, and 10
# layers = [10, 20, 10]
# connectivities = [
#     [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
#     [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
#     [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
# ]
# visualize_neural_network(connectivities)
