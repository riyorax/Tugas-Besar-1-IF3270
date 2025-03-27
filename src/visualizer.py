import numpy as np
import json
import os
import datetime

def _create_node(node_id, label, level, group, title):
    return {
        'id': node_id,
        'label': label,
        'level': level,
        'group': group,
        'title': title
    }

def _create_edge(edgeId, from_id, to_id, title, color, width, dashes=False):
    return {
        'id': edgeId,
        'from': from_id,
        'to': to_id, 
        'title': title,
        'color': color,
        'width': width,
        'dashes': dashes
    }

def adjust_color_brightness(hex_color, factor):

    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    
    r = min(255, max(0, r))
    g = min(255, max(0, g))
    b = min(255, max(0, b))
    
    return f'#{r:02x}{g:02x}{b:02x}'

def visualize_ann(model, input_shape, filename='ann', show_bias=True, css_path=None, js_path=None, output_dir=None):
  
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    
    if (css_path is None):
        css_path = os.path.join('..','static','style.css')
    if (js_path is None):
      js_path =  os.path.join('..','static','script.js')
    
    if isinstance(input_shape, tuple):
        if len(input_shape) == 1:
            input_neurons = input_shape[0]
        else:
            input_neurons = np.prod(input_shape)
    else:
        input_neurons = input_shape
    
    nodes = []
    edges = []
    node_id_counter = 0
    
    input_layer_ids = []
    for i in range(min(input_neurons, 10)):
        node_id = node_id_counter
        nodes.append(_create_node(node_id, f'I{i}', 0, 'input', f'Input neuron {i}'))
        input_layer_ids.append(node_id)
        node_id_counter += 1
    
    if input_neurons > 10:
        node_id = node_id_counter
        nodes.append(_create_node(node_id, f'... ({input_neurons-10} more)', 0, 'input', 'Additional input neurons'))
        input_layer_ids.append(node_id)
        node_id_counter += 1
    
    prev_layer_ids = input_layer_ids
    bias_node_ids = []
    
    layer_colors = ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099', '#0099C6', '#DD4477', '#66AA00']
    
    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer, 'output_size'):
            continue
        
        activation_name = "None"
        if hasattr(layer, 'activation') and layer.activation is not None:
            if hasattr(layer.activation, 'get_name'):
                activation_name = layer.activation.get_name()
        
        layer_color = layer_colors[layer_idx % len(layer_colors)]
        
        current_layer_ids = []
        for i in range(min(layer.output_size, 10)):
            node_id = node_id_counter
            nodes.append(_create_node(
                node_id,
                f'L{layer_idx}_{i}', 
                layer_idx + 1, 
                'hidden' if layer_idx < len(model.layers) - 1 else 'output', 
                f'Layer {layer_idx+1}, Neuron {i}<br>Activation: {activation_name}'
            ))
            current_layer_ids.append(node_id)
            node_id_counter += 1
        
        if layer.output_size > 10:
            node_id = node_id_counter
            nodes.append(_create_node(
                node_id, 
                f'... ({layer.output_size-10} more)', 
                layer_idx + 1, 
                'hidden' if layer_idx < len(model.layers) - 1 else 'output', 
                f'Additional neurons in layer {layer_idx+1}'
            ))
            current_layer_ids.append(node_id)
            node_id_counter += 1
        
        if show_bias and hasattr(layer, 'biases') and layer.biases is not None:
            bias_node_id = node_id_counter
            nodes.append(_create_node(
                bias_node_id,
                f'B{layer_idx}',
                layer_idx + 1,
                'bias',
                f'Bias for layer {layer_idx+1}'
            ))
            bias_node_ids.append(bias_node_id)
            node_id_counter += 1
            
            bias_data = layer.biases.data if hasattr(layer.biases, 'data') else layer.biases
            
            for j, to_id in enumerate(current_layer_ids):
                if j < 10:
                    bias_val = bias_data[0, j] if j < bias_data.shape[1] else 0
                    bias_text = f"Bias: {bias_val:.4f}"
                    
                    edge_id = f"bias_edge_{bias_node_id}_{to_id}"
                    
                    bias_color = adjust_color_brightness(layer_color, 0.7)
                    edges.append(_create_edge(
                        edge_id, 
                        bias_node_id, 
                        to_id,
                        bias_text, 
                        bias_color, 
                        1.0,
                        dashes=True
                    ))
                else:
                    more_node = [n for n in current_layer_ids if nodes[n]['label'].startswith('...')]
                    if more_node:
                        edge_id = f"bias_edge_{bias_node_id}_{more_node[0]}"
                        edges.append(_create_edge(
                            edge_id,
                            bias_node_id,
                            more_node[0],
                            "Multiple bias connections",
                            adjust_color_brightness(layer_color, 0.7),
                            1.0,
                            dashes=True
                        ))
        
        if hasattr(layer, 'weights') and layer.weights is not None:
            weight_data = layer.weights.data if hasattr(layer.weights, 'data') else layer.weights
            weight_shape = weight_data.shape
            
            for i, from_id in enumerate(prev_layer_ids):
                from_is_ellipsis = nodes[from_id]['label'].startswith('...')
                
                for j, to_id in enumerate(current_layer_ids):
                    to_is_ellipsis = nodes[to_id]['label'].startswith('...')
                    
                    if not from_is_ellipsis and not to_is_ellipsis:
                        from_idx = int(nodes[from_id]['label'].split('_')[-1]) if '_' in nodes[from_id]['label'] else int(nodes[from_id]['label'][1:])
                        to_idx = int(nodes[to_id]['label'].split('_')[-1]) if '_' in nodes[to_id]['label'] else 0
                        
                        if from_idx < weight_shape[0] and to_idx < weight_shape[1]:
                            weight_val = weight_data[from_idx, to_idx]
                            weight_text = f"Weight: {weight_val:.4f}"
                            
                            edge_id = f"edge_{from_id}_{to_id}"
                            
                            edges.append(_create_edge(
                                edge_id, 
                                from_id, 
                                to_id, 
                                weight_text, 
                                layer_color,
                                1.0
                            ))
                    else:
                        edge_id = f"edge_{from_id}_{to_id}"
                        
                        edges.append(_create_edge(
                            edge_id,
                            from_id, 
                            to_id,
                            'Various Connections', 
                            layer_color, 
                            1.0
                        ))
        
        prev_layer_ids = current_layer_ids
    
    network_data = {
        'nodes': nodes,
        'edges': edges
    }
    
    loss_fn = getattr(model, 'loss', 'Unknown')

    html_path = create_html_file(network_data, loss_fn, 
        filename, css_path, js_path, output_dir)
    
    return html_path

def create_html_file(network_data, loss_fn, filename, css_path, js_path,output_dir=None):
  
    network_json = json.dumps(network_data)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Visualization</title>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis-network.min.css" rel="stylesheet" type="text/css" />
    <link href="{css_path}" rel="stylesheet" type="text/css" />
    <script type="text/javascript">
        const networkData = {network_json};
    </script>
    <script src="{js_path}"></script>
</head>
<body>
    <div class="info">
        <strong>Loss Function:</strong> {loss_fn}
    </div>
    <div id="network"></div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background-color: #3366CC;"></div>
            <div class="legend-label">Weight Connections</div>
        </div>
        <div class="legend-item">
            <div class="legend-color dashed"></div>
            <div class="legend-label">Bias Connections</div>
        </div>
    </div>
</body>
</html>
"""
    if (output_dir is None):
        output_dir = os.path.join("ann_visualizer", "output")
        
    html_path = os.path.join(output_dir, f"{filename}.html")
    
    with open(html_path, "w") as f:
        f.write(html)
    return html_path

if __name__ == "__main__":
    try:
        from ann import NeuralNetwork
        from dense_layer import DenseLayer
        from activations import sigmoid, relu, tanh
        
        model = NeuralNetwork('mse')
        model.add_layer(DenseLayer(3, sigmoid))
        model.add_layer(DenseLayer(3, relu))
        model.add_layer(DenseLayer(9, tanh))
        
        dummy_input = np.random.randn(10, 4)
        model.forward(dummy_input)
        
        visualize_ann(model, 4)
    except ImportError as e:
        print(f"Error importing required modules: {e}")