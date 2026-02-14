#!/usr/bin/env python3
"""
Simple Gradio viewer for point cloud PLY files
"""
import gradio as gr
import numpy as np
from pathlib import Path

def load_ply(ply_path: str):
    """Load PLY file and return as numpy arrays"""
    points = []
    colors = []
    
    with open(ply_path, 'r') as f:
        # Skip header
        line = f.readline()
        while 'end_header' not in line:
            line = f.readline()
        
        # Read points
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                points.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def view_pointcloud(ply_file):
    """Convert PLY to plotly 3D scatter"""
    import plotly.graph_objects as go
    
    if ply_file is None:
        return None
    
    points, colors = load_ply(ply_file)
    
    # Subsample if too many points
    max_points = 100000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    
    # Convert colors to plotly format
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors]
    
    # Create 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=color_strings,
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (Depth)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Point Cloud Viewer"
    )
    
    return fig

# Create Gradio interface
with gr.Blocks(title="Point Cloud Viewer") as demo:
    gr.Markdown("# 🌐 Point Cloud Viewer")
    gr.Markdown("Upload a PLY file or use the default from DA3 output")
    
    with gr.Row():
        ply_input = gr.File(
            label="PLY File",
            file_types=[".ply"],
            value="/home/hernan/stac/scene/da3_output/pointcloud.ply"
        )
    
    with gr.Row():
        view_btn = gr.Button("View Point Cloud", variant="primary")
    
    with gr.Row():
        plot_output = gr.Plot(label="3D Point Cloud")
    
    view_btn.click(
        fn=view_pointcloud,
        inputs=[ply_input],
        outputs=[plot_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
