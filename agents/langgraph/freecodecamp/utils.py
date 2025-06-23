import os
from langgraph.graph.graph import CompiledGraph

def convert_to_png(graph: CompiledGraph, image_name: str = "graph") -> None:
    try:
        png_graph = graph.get_graph().draw_mermaid_png()
        path = os.path.join("graph-images",f"{image_name}.png")
        with open(path, "wb") as f:
            f.write(png_graph)
            print(f"Graph saved as {image_name}.png in {path}")
    except Exception as e:
        print(f"Exception: {e}")