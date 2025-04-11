from pathlib import Path


from linksac import PROJECT_DIR
from linksac import simulations
from linksac import datasets

def test_edge_anomaly_injection():
    data_dir = Path(PROJECT_DIR, "data")
    graph = datasets.load_wikipedia(data_dir)

    injections = {
        "Random": lambda g: simulations.inject_random_edge_anomalies(g, 0.1),
        "Embedding": lambda g: simulations.inject_embedding_edge_anomalies(g, 0.1, 50),
        "Attack": lambda g: simulations.inject_attack_edge_anomalies(g, 0.1)
    }

    for name, func in injections.items():
        func(graph)