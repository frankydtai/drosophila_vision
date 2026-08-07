import neuro_framework.connectome
from neuro_framework.connectome import ConnectomeLoader

print(neuro_framework.connectome.__file__)

def main() -> None:
    loader = ConnectomeLoader.from_banc(    
        cell_types=["T4c"]
    )
    nodes, edges = loader.load()
    print(loader.summary())
    print(nodes.head())
    print(edges.head())


if __name__ == "__main__":
    main()
