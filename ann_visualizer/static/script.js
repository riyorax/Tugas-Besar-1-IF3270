function initializeVisualization(networkData) {
  const container = document.getElementById('network');

  networkData.edges.forEach(edge => {
    edge.originalColor = edge.color;
    edge.originalDashes = edge.dashes;
  });

  const nodes = new vis.DataSet(networkData.nodes);
  const edges = new vis.DataSet(networkData.edges);

  const data = {
    nodes: nodes,
    edges: edges
  };

  const options = {
    layout: {
      hierarchical: {
        direction: 'LR',
        sortMethod: 'directed',
        levelSeparation: 150,
        nodeSpacing: 120
      }
    },
    nodes: {
      shape: 'circle',
      size: 25,
      font: {
        size: 14
      },
      borderWidth: 2,
      shadow: true,
      color: {
        border: '#2B7CE9',
        background: '#97C2FC',
        highlight: {
          border: '#2B7CE9',
          background: '#D2E5FF'
        }
      },
      groups: {
        input: {
          color: { background: 'lightblue', border: '#2B7CE9' },
          label: 'Input Layer'
        },
        hidden: {
          color: { background: '#FFFF99', border: '#FFA500' },
          label: 'Hidden Layer'
        },
        output: {
          color: { background: '#FF9999', border: '#FF0000' },
          label: 'Output Layer'
        },
        bias: {
          color: { background: '#E6E6E6', border: '#666666' },
          shape: 'box',
          label: 'Bias'
        }
      }
    },
    edges: {
      smooth: {
        type: 'cubicBezier',
        forceDirection: 'horizontal'
      },
      arrows: {
        to: { enabled: true, scaleFactor: 0.5 }
      },
      selectionWidth: 3,
      width: 1.0
    },
    physics: {
      hierarchicalRepulsion: {
        nodeDistance: 120
      },
      stabilization: {
        iterations: 100
      }
    },
    interaction: {
      // hover: true,
      tooltipDelay: 200
    }
  };

  const network = new vis.Network(container, data, options);

  network.on("selectNode", function (params) {
    const nodeId = params.nodes[0];

    const connectedEdges = network.getConnectedEdges(nodeId);

    connectedEdges.forEach(edgeId => {
      const edge = edges.get(edgeId);
      if (edge) {
        const glowColor = edge.originalColor || "#FFFF00";

        edges.update({
          id: edgeId,
          width: 3,
          shadow: true,
          shadowColor: glowColor,
          shadowSize: 10,
          shadowX: 0,
          shadowY: 0,
          color: glowColor,
          dashes: edge.originalDashes
        });
      }
    });
  });


  network.on("deselectNode", function () {
    const allEdges = edges.getIds();

    allEdges.forEach(edgeId => {
      const edge = edges.get(edgeId);
      if (edge) {
        edges.update({
          id: edgeId,
          width: 1.0,
          shadow: false,
          color: edge.originalColor,
          dashes: edge.originalDashes
        });
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', function () {
  if (typeof networkData !== 'undefined') {
    initializeVisualization(networkData);
  } else {
    console.error("Network data not found");
  }
});