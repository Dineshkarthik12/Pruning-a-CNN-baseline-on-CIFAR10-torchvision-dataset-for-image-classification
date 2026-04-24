const resultsData = [
    { label: "None", lambda: "0.0", acc: 81.81, sparsity: 1.90 },
    { label: "Low", lambda: "1e-4", acc: 81.10, sparsity: 88.62 },
    { label: "Medium", lambda: "1e-3", acc: 81.73, sparsity: 96.19 },
    { label: "High", lambda: "1e-2", acc: 72.33, sparsity: 99.64 },
    { label: "Very High", lambda: "5e-2", acc: 30.54, sparsity: 99.99 }
];

const canvas = document.getElementById('network-canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('lambda-slider');

const accVal = document.getElementById('accuracy-val');
const sparsityVal = document.getElementById('sparsity-val');
const connVal = document.getElementById('connections-val');

// Network Architecture for Visualization
const layers = [20, 16, 10]; 
const nodes = [];
const connections = [];

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    initNetwork();
    drawNetwork(resultsData[slider.value].sparsity);
}

window.addEventListener('resize', resizeCanvas);

function initNetwork() {
    nodes.length = 0;
    connections.length = 0;

    const layerSpacing = canvas.width / (layers.length + 1);

    // Create Nodes
    layers.forEach((nodeCount, layerIndex) => {
        const x = layerSpacing * (layerIndex + 1);
        const ySpacing = (canvas.height * 0.8) / nodeCount;
        const startY = (canvas.height - (ySpacing * (nodeCount - 1))) / 2;

        const layerNodes = [];
        for (let i = 0; i < nodeCount; i++) {
            layerNodes.push({
                x: x,
                y: startY + i * ySpacing,
                layer: layerIndex
            });
        }
        nodes.push(layerNodes);
    });

    // Create Connections
    for (let l = 0; l < layers.length - 1; l++) {
        const currentLayer = nodes[l];
        const nextLayer = nodes[l + 1];

        currentLayer.forEach(nodeA => {
            nextLayer.forEach(nodeB => {
                connections.push({
                    start: nodeA,
                    end: nodeB,
                    importance: Math.random() // Random importance score for pruning simulation
                });
            });
        });
    }

    // Sort connections by importance so we can prune the lowest % accurately
    connections.sort((a, b) => a.importance - b.importance);
}

function drawNetwork(sparsityPercent) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const totalConnections = connections.length;
    const connectionsToPrune = Math.floor(totalConnections * (sparsityPercent / 100));
    const activeConnectionsCount = totalConnections - connectionsToPrune;

    // Draw Connections
    connections.forEach((conn, index) => {
        const isPruned = index < connectionsToPrune;
        
        ctx.beginPath();
        ctx.moveTo(conn.start.x, conn.start.y);
        ctx.lineTo(conn.end.x, conn.end.y);
        
        if (isPruned) {
            // Draw faintly if we want to show it, or skip drawing completely
            // Skipping drawing looks cleaner for high sparsity
            if (sparsityPercent < 90) {
                ctx.strokeStyle = 'rgba(51, 65, 85, 0.1)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        } else {
            // Active connection
            // Calculate opacity based on importance to give a glowing effect
            const importanceFactor = (index - connectionsToPrune) / activeConnectionsCount;
            const opacity = 0.3 + (importanceFactor * 0.7);
            
            ctx.strokeStyle = `rgba(16, 185, 129, ${opacity})`;
            ctx.lineWidth = 1.5;
            
            // Add slight glow
            ctx.shadowBlur = 5;
            ctx.shadowColor = 'rgba(16, 185, 129, 0.5)';
            ctx.stroke();
            ctx.shadowBlur = 0; // Reset
        }
    });

    // Draw Nodes
    nodes.forEach((layerNodes, layerIndex) => {
        layerNodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 4, 0, Math.PI * 2);
            
            // Nodes are active if they have at least one active connection,
            // but for simplicity we'll just style them based on general layer activity
            if (sparsityPercent > 99 && layerIndex > 0) {
                 ctx.fillStyle = '#334155'; // Inactive
                 ctx.shadowBlur = 0;
            } else {
                ctx.fillStyle = '#10b981'; // Active
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#10b981';
            }
            
            ctx.fill();
            ctx.shadowBlur = 0; // Reset
        });
    });

    // Update UI Metrics
    const data = resultsData[slider.value];
    
    // Animation for numbers
    animateValue(accVal, parseFloat(accVal.innerText), data.acc, 500, '%');
    animateValue(sparsityVal, parseFloat(sparsityVal.innerText) || 0, data.sparsity, 500, '%');
    
    // Real calculated connections vs theoretical
    const theoreticalTotal = 24*16*10; // based on simple math if we wanted a bigger display
    connVal.innerText = activeConnectionsCount;
}

function animateValue(obj, start, end, duration, suffix = '') {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = (progress * (end - start) + start).toFixed(2);
        obj.innerHTML = current + suffix;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end.toFixed(2) + suffix;
            // Update color based on values
            if (obj === accVal) {
                if (end < 50) obj.style.color = '#ef4444'; // red
                else if (end < 80) obj.style.color = '#f59e0b'; // yellow
                else obj.style.color = '#10b981'; // green
            }
        }
    };
    window.requestAnimationFrame(step);
}

slider.addEventListener('input', (e) => {
    const data = resultsData[e.target.value];
    drawNetwork(data.sparsity);
});

// Initial Setup
setTimeout(() => {
    resizeCanvas();
    // initial draw trigger
    slider.dispatchEvent(new Event('input'));
}, 100);
