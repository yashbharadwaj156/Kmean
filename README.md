import random
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision
import logging
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for the create_clients function
def create_clients(X_train, y_train, nb_classes, sampling_technique, num_clients, initial):
    logging.info("Creating clients...")
    clients = {}
    for i in range(num_clients):
        clients[f'{initial}_{i}'] = list(zip(X_train[i::num_clients], y_train[i::num_clients]))
    logging.info(f"{num_clients} clients created.")
    return clients

# Placeholder for the batch_data function
def batch_data(data, BATCH_SIZE):
    logging.info("Batching data...")
    data, labels = zip(*data)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(labels))).batch(BATCH_SIZE)
    logging.info("Data batched.")
    return dataset

# Placeholder for the get_model function
def get_model(input_shape, nb_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(nb_classes, activation='softmax')
    ])
    return model

# Visualization function
def visualize_federated_learning_process(client_set, cluster_heads, client_clusters, global_model=None, initial=False):
    G = nx.DiGraph()
    for client in client_set.keys():
        G.add_node(client, color='lightblue')
    
    for head in cluster_heads.values():
        G.add_node(head, color='orange')
    
    G.add_node('Global Server', color='red')
    for client, head in zip(client_set.keys(), [cluster_heads[client_clusters[list(client_set.keys()).index(client)]] for client in client_set.keys()]):
        G.add_edge(client, head)
    
    for head in cluster_heads.values():
        G.add_edge(head, 'Global Server')
    
    node_colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    
    for client in client_set.keys():
        x, y = pos[client]
        if initial:
            weight = 0
        else:
            weight = client_set[client]['model'].get_weights()[0].flatten()[0]
        plt.text(x, y + 0.1, s=f'{weight:.2f}', bbox=dict(facecolor='lightblue', alpha=0.5), horizontalalignment='center')
    
    for head in cluster_heads.values():
        x, y = pos[head]
        if initial:
            weight = 0
        else:
            weight = client_set[head]['model'].get_weights()[0].flatten()[0]
        plt.text(x, y + 0.1, s=f'{weight:.2f}', bbox=dict(facecolor='orange', alpha=0.5), horizontalalignment='center')
    
    if global_model is not None:
        global_server_weight = global_model.get_weights()[0].flatten()[0]
    else:
        global_server_weight = 0
    x, y = pos['Global Server']
    plt.text(x, y + 0.1, s=f'{global_server_weight:.2f}', bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
    
    plt.title('Federated Learning Communication Flow')
    plt.show()

# Initialize and batch clients
clients_batched = create_clients(X_train, y_train, nb_classes, sampling_technique, num_clients=10, initial='client')
logging.info("Clients created and batched.")

del X_train, y_train

# Function to perform clustering
def perform_clustering(client_set):
    # Compute mean feature vectors for each client
    client_features = [np.mean(np.array([x for x, _ in client_set[client]['dataset']]), axis=0).flatten() for client in client_set.keys()]

    # Standardize the features
    scaler = StandardScaler()
    client_features = scaler.fit_transform(client_features)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    client_clusters = kmeans.fit_predict(client_features)
    logging.info("K-means clustering performed on client data.")

    # Assign cluster heads
    cluster_heads = {}
    clusters = defaultdict(list)
    for client_name, cluster in zip(client_set.keys(), client_clusters):
        clusters[cluster].append(client_name)
    for cluster, members in clusters.items():
        cluster_heads[cluster] = members[0]
    logging.info("Cluster heads assigned.")
    for cluster, head in cluster_heads.items():
        logging.info(f"Cluster {cluster}: Head {head}")
    
    return client_clusters, cluster_heads

# Perform initial clustering
client_clusters, cluster_heads = perform_clustering(clients_batched)

# Visualize initial clustering with initial weights
visualize_federated_learning_process(clients_batched, cluster_heads, client_clusters, initial=True)

# Process and batch the training data for each client
client_set = {k: {} for k in clients_batched.keys()}
for (client_name, data) in clients_batched.items():
    client_set[client_name]["dataset"] = batch_data(data, BATCH_SIZE)
    local_model = get_model(input_shape, nb_classes)
    local_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(), Precision(), 'accuracy'])
    client_set[client_name]["model"] = local_model
    logging.info(f"Model for {client_name} initialized and compiled.")

# Training loop
comms_round = 10
global_model = get_model(input_shape, nb_classes)
global_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(), Precision(), 'accuracy'])

for comm_round in range(comms_round):
    logging.info(f"Communication round {comm_round + 1}")
    global_weights = global_model.get_weights()
    scaled_local_weight_list = []

    client_names = list(client_set.keys())
    random.shuffle(client_names)

    for client in client_names:
        logging.info(f"Training client: {client}")
        scaled_weights = train_client(client, global_weights, class_weights, comm_round)
        scaled_local_weight_list.append(scaled_weights)

    # Send updates to cluster heads
    cluster_updates = send_updates_to_cluster_head(client_set, cluster_heads)
    logging.info("Updates sent to cluster heads.")

    # Cluster heads aggregate updates
    cluster_heads_weights = aggregate_cluster_updates(cluster_updates)
    logging.info("Cluster heads aggregated updates.")

    # Global model aggregates from cluster heads
    global_weights = global_aggregation(cluster_heads_weights)
    global_model.set_weights(global_weights)
    logging.info("Global model aggregated updates from cluster heads.")

    # Test global model and print out metrics after each communication round
    g_accuracy = 0
    for (x_batch, y_batch) in test_batched:
        g_loss, g_accuracy, g_precision, g_recall, g_f1 = test_model(x_batch, y_batch, global_model, comm_round)
        global_loss.append(g_loss)
        global_accuracy.append(g_accuracy)
        global_precision.append(g_precision)
        global_recall.append(g_recall)
        global_f1.append(g_f1)

    if g_accuracy > best_global_accuracy:
        best_global_accuracy = g_accuracy
        global_model.save_weights('global_model_best_weights.h5')
        logging.info("New Weights Saved")

    # Re-cluster clients based on updated weights
    client_clusters, cluster_heads = perform_clustering(client_set)
    
    # Visualize the process after each communication round
    visualize_federated_learning_process(client_set, cluster_heads, client_clusters, global_model)
