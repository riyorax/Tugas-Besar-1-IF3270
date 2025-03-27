function initializeVisualization(networkData) {
  const container = document.getElementById('network');

  let selectedEdgeId = null;
  const weightInfoDiv = document.createElement('div');
  weightInfoDiv.id = 'weight-info';
  weightInfoDiv.className = 'weight-info';
  weightInfoDiv.style.display = 'none';
  document.body.appendChild(weightInfoDiv);

  networkData.edges.forEach(edge => {
    edge.originalColor = edge.color;
    edge.originalDashes = edge.dashes;
    edge.originalWidth = edge.width || 1.0;
    edge.weightValue = edge.title;
    delete edge.title;
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
      hover: false,
      hoverConnectedEdges: false,
      selectable: true,
      selectConnectedEdges: false,
      tooltipDelay: 200,
      hideEdgesOnDrag: false,
      hideEdgesOnZoom: false,
      tooltipDelay: null
    }
  };

  const network = new vis.Network(container, data, options);
  
  function resetAllEdges() {
    const allEdges = edges.getIds();

    allEdges.forEach(edgeId => {
      const edge = edges.get(edgeId);
      if (edge) {
        edges.update({
          id: edgeId,
          width: edge.originalWidth,
          shadow: false,
          color: edge.originalColor,
          dashes: edge.originalDashes
        });
      }
    });
    
    weightInfoDiv.style.display = 'none';
    selectedEdgeId = null;
  }

  network.on("selectEdge", function (params) {
    resetAllEdges();
    
    if (params.edges.length === 0) {
      return;
    }
    
    const edgeId = params.edges[0];
    selectedEdgeId = edgeId;
    
    const edge = edges.get(edgeId);
    if (!edge) return;
    const highlightColor = '#FF9900';
    
    edges.update({
      id: edgeId,
      width: 3,
      shadow: true,
      shadowColor: highlightColor,
      shadowSize: 10,
      shadowX: 0,
      shadowY: 0,
      color: highlightColor
    });
    
    if (edge.weightValue && edge.weightValue !== 'Various Connections') {
      displayWeightInfo(edge.weightValue, network.getPositions([edge.from, edge.to]));
    } else {
      displayWeightInfo("Multiple connections with different weights",  network.getPositions([edge.from, edge.to]));
    }
  });
  function displayWeightInfo(text, positions) {
    const fromPos = positions[Object.keys(positions)[0]];
    const toPos = positions[Object.keys(positions)[1]];
    
    const middleX = (fromPos.x + toPos.x) / 2;
    const middleY = (fromPos.y + toPos.y) / 2;
    
    weightInfoDiv.innerHTML = text;
    weightInfoDiv.style.display = 'block';
    
    const domPos = network.canvasToDOM({ x: middleX, y: middleY });
    
    weightInfoDiv.style.left = (domPos.x + 10) + 'px';
    weightInfoDiv.style.top = (domPos.y + 10) + 'px';
  }
  
  network.on("click", function (params) {
    if (params.nodes.length === 0 && params.edges.length === 0) {
      resetAllEdges();
      network.unselectAll();
    }
  });

  network.on("selectNode", function (params) {
    const nodeId = params.nodes[0];
    
    if (selectedEdgeId) {
      const edge = edges.get(selectedEdgeId);
      if (edge) {
        edges.update({
          id: selectedEdgeId,
          width: edge.originalWidth,
          shadow: false,
          color: edge.originalColor,
          dashes: edge.originalDashes
        });
      }
      weightInfoDiv.style.display = 'none';
      selectedEdgeId = null;
    }

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

  network.on("deselectNode", resetAllEdges);
  
  window.addEventListener('resize', function() {
    if (selectedEdgeId && weightInfoDiv.style.display === 'block') {
      const edge = edges.get(selectedEdgeId);
      if (edge) {
        const positions = network.getPositions([edge.from, edge.to]);
        displayWeightInfo(weightInfoDiv.innerHTML, positions);
      }
    }
  });
  
  const legend = document.querySelector('.legend');
  if (legend) {
    const legendInfo = document.createElement('div');
    legendInfo.className = 'legend-info';
    legendInfo.innerHTML = '<p>Click on an edge to see its weight value. Click on a node to highlight all its connections.</p>';
    legend.appendChild(legendInfo);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  if (typeof networkData !== 'undefined') {
    initializeVisualization(networkData);
  } else {
    console.error("Network data not found");
  }
});