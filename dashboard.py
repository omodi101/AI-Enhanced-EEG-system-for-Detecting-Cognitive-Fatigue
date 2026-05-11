"""
Real-Time Interactive Dashboard
Live monitoring interface for cognitive fatigue detection
"""

import numpy as np
import json
from datetime import datetime
import time


def create_dashboard_html(output_path: str = '/home/claude/dashboard.html'):
    """
    Create interactive HTML dashboard with live monitoring capabilities.
    Uses real-time updates and WebSocket-style visualization.
    """
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Fatigue Monitor - Real-Time Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .card-title {
            font-size: 1.4em;
            font-weight: 600;
            color: #333;
        }
        
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            animation: pulse 2s infinite;
        }
        
        .status-focused {
            background: #d4edda;
            color: #155724;
        }
        
        .status-fatigued {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-unknown {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .metric-display {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        
        .metric {
            text-align: center;
            flex: 1;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        button {
            flex: 1;
            min-width: 120px;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c0392b;
        }
        
        #brainwaveChart, #timelineChart, #performanceChart, #bandPowerChart {
            width: 100%;
            height: 350px;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .alert {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .alert-warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        
        .alert-danger {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .alert-info {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }
        
        .log-container {
            max-height: 200px;
            overflow-y: auto;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }
        
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .timestamp {
            color: #666;
            margin-right: 10px;
        }
        
        .recommendation-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
        }
        
        .recommendation-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .recommendation-list {
            list-style: none;
        }
        
        .recommendation-list li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        .recommendation-list li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }
        
        footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Cognitive Fatigue Monitor</h1>
            <p class="subtitle">AI-Enhanced Real-Time EEG Analysis System</p>
        </header>
        
        <div id="alertContainer"></div>
        
        <div class="dashboard-grid">
            <!-- Current State Card -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Current State</h2>
                    <span id="statusBadge" class="status-badge status-unknown">INITIALIZING</span>
                </div>
                <div class="metric-display">
                    <div class="metric">
                        <div class="metric-value" id="confidenceValue">--</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="latencyValue">--</div>
                        <div class="metric-label">Latency (ms)</div>
                    </div>
                </div>
                <div class="controls">
                    <button class="btn-primary" onclick="startMonitoring()">Start Monitor</button>
                    <button class="btn-secondary" onclick="pauseMonitoring()">Pause</button>
                    <button class="btn-danger" onclick="resetMonitoring()">Reset</button>
                </div>
            </div>
            
            <!-- Performance Metrics Card -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Session Metrics</h2>
                </div>
                <div class="metric-display">
                    <div class="metric">
                        <div class="metric-value" id="focusTime">0</div>
                        <div class="metric-label">Focused (min)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="fatigueTime">0</div>
                        <div class="metric-label">Fatigued (min)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="totalPredictions">0</div>
                        <div class="metric-label">Predictions</div>
                    </div>
                </div>
                <div id="performanceChart"></div>
            </div>
            
            <!-- Live Brainwave Simulation -->
            <div class="card full-width">
                <div class="card-header">
                    <h2 class="card-title">Real-Time EEG Signal</h2>
                    <span id="signalQuality" style="color: #28a745; font-weight: 600;">● GOOD</span>
                </div>
                <div id="brainwaveChart"></div>
            </div>
            
            <!-- Frequency Band Powers -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Frequency Band Analysis</h2>
                </div>
                <div id="bandPowerChart"></div>
            </div>
            
            <!-- State Timeline -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Cognitive State Timeline</h2>
                </div>
                <div id="timelineChart"></div>
            </div>
            
            <!-- Recommendations -->
            <div class="card full-width">
                <div class="card-header">
                    <h2 class="card-title">AI Recommendations</h2>
                </div>
                <div id="recommendationsContainer" class="recommendation-box">
                    <div class="recommendation-title">📊 Waiting for data...</div>
                    <ul class="recommendation-list">
                        <li>Start monitoring to receive personalized recommendations</li>
                    </ul>
                </div>
            </div>
            
            <!-- Activity Log -->
            <div class="card full-width">
                <div class="card-header">
                    <h2 class="card-title">Activity Log</h2>
                </div>
                <div class="log-container" id="logContainer">
                    <div class="log-entry">
                        <span class="timestamp">[00:00:00]</span>
                        <span>System initialized. Ready to monitor.</span>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p><strong>ISEF Science Fair Project</strong></p>
            <p>AI-Enhanced Low-Cost EEG System for Real-Time Detection of Cognitive Fatigue Using Edge Computing</p>
            <p style="margin-top: 10px; opacity: 0.8;">Demo Mode - Simulated EEG Data</p>
        </footer>
    </div>
    
    <script>
        // Global state
        let monitoringActive = false;
        let sessionStart = null;
        let predictionHistory = [];
        let focusedSeconds = 0;
        let fatiguedSeconds = 0;
        let totalPredictions = 0;
        let brainwaveData = {
            time: [],
            signal: [],
            maxPoints: 100
        };
        let bandPowers = {
            delta: [],
            theta: [],
            alpha: [],
            beta: []
        };
        
        // Initialize charts
        function initializeCharts() {
            // Brainwave chart
            Plotly.newPlot('brainwaveChart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#667eea', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(102, 126, 234, 0.1)'
            }], {
                title: '',
                xaxis: { title: 'Time (s)', range: [0, 10] },
                yaxis: { title: 'Amplitude (μV)' },
                margin: { t: 10, r: 10, b: 40, l: 50 }
            }, {responsive: true});
            
            // Timeline chart
            Plotly.newPlot('timelineChart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#764ba2', width: 3 },
                marker: { size: 8 }
            }], {
                title: '',
                xaxis: { title: 'Time' },
                yaxis: { 
                    title: 'State',
                    tickvals: [0, 1],
                    ticktext: ['Focused', 'Fatigued']
                },
                margin: { t: 10, r: 10, b: 40, l: 60 }
            }, {responsive: true});
            
            // Performance chart
            Plotly.newPlot('performanceChart', [{
                values: [1, 1],
                labels: ['Focused', 'Fatigued'],
                type: 'pie',
                marker: {
                    colors: ['#2ecc71', '#e74c3c']
                },
                textinfo: 'label+percent',
                hole: 0.4
            }], {
                title: '',
                margin: { t: 10, r: 10, b: 10, l: 10 },
                showlegend: false
            }, {responsive: true});
            
            // Band power chart
            Plotly.newPlot('bandPowerChart', [{
                x: ['Delta', 'Theta', 'Alpha', 'Beta'],
                y: [0, 0, 0, 0],
                type: 'bar',
                marker: {
                    color: ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
                }
            }], {
                title: '',
                xaxis: { title: 'Frequency Band' },
                yaxis: { title: 'Power' },
                margin: { t: 10, r: 10, b: 40, l: 50 }
            }, {responsive: true});
        }
        
        // Simulate EEG data
        function simulateEEGSample(state = 'focused') {
            const t = brainwaveData.time.length * 0.1;
            
            let beta, alpha, theta;
            if (state === 'focused') {
                beta = 8 * Math.sin(2 * Math.PI * 17 * t);
                alpha = 5 * Math.sin(2 * Math.PI * 10 * t);
                theta = 2 * Math.sin(2 * Math.PI * 6 * t);
            } else {
                beta = 3 * Math.sin(2 * Math.PI * 17 * t);
                alpha = 7 * Math.sin(2 * Math.PI * 10 * t);
                theta = 6 * Math.sin(2 * Math.PI * 6 * t);
            }
            
            const signal = beta + alpha + theta + (Math.random() - 0.5) * 2;
            return signal;
        }
        
        // Make prediction (simulated)
        function makePrediction() {
            // Simulate prediction
            const isFatigued = Math.random() > 0.65;
            const confidence = 0.75 + Math.random() * 0.2;
            const latency = 20 + Math.random() * 30;
            
            // Simulate band powers
            const bandPowerValues = {
                delta: Math.random() * 50 + 20,
                theta: isFatigued ? Math.random() * 80 + 60 : Math.random() * 40 + 20,
                alpha: Math.random() * 70 + 40,
                beta: isFatigued ? Math.random() * 40 + 20 : Math.random() * 80 + 60
            };
            
            return {
                state: isFatigued ? 'fatigued' : 'focused',
                confidence: confidence,
                latency: latency,
                bandPowers: bandPowerValues
            };
        }
        
        // Update dashboard
        function updateDashboard(prediction) {
            totalPredictions++;
            
            // Update status badge
            const badge = document.getElementById('statusBadge');
            if (prediction.state === 'focused') {
                badge.textContent = 'FOCUSED';
                badge.className = 'status-badge status-focused';
                focusedSeconds++;
            } else {
                badge.textContent = 'FATIGUED';
                badge.className = 'status-badge status-fatigued';
                fatiguedSeconds++;
                
                // Show alert if fatigue detected
                if (Math.random() > 0.7) {
                    showAlert('Fatigue detected! Consider taking a break.', 'warning');
                }
            }
            
            // Update metrics
            document.getElementById('confidenceValue').textContent = 
                (prediction.confidence * 100).toFixed(0) + '%';
            document.getElementById('latencyValue').textContent = 
                prediction.latency.toFixed(1);
            document.getElementById('focusTime').textContent = 
                (focusedSeconds / 60).toFixed(1);
            document.getElementById('fatigueTime').textContent = 
                (fatiguedSeconds / 60).toFixed(1);
            document.getElementById('totalPredictions').textContent = totalPredictions;
            
            // Update performance pie chart
            Plotly.restyle('performanceChart', {
                values: [[focusedSeconds, fatiguedSeconds]]
            });
            
            // Update band powers
            Plotly.restyle('bandPowerChart', {
                y: [[
                    prediction.bandPowers.delta,
                    prediction.bandPowers.theta,
                    prediction.bandPowers.alpha,
                    prediction.bandPowers.beta
                ]]
            });
            
            // Update timeline
            predictionHistory.push({
                time: new Date(),
                state: prediction.state === 'focused' ? 0 : 1
            });
            
            if (predictionHistory.length > 50) {
                predictionHistory.shift();
            }
            
            const timeLabels = predictionHistory.map((p, i) => i);
            const stateValues = predictionHistory.map(p => p.state);
            
            Plotly.restyle('timelineChart', {
                x: [timeLabels],
                y: [stateValues]
            });
            
            // Update recommendations
            updateRecommendations(prediction);
            
            // Add log entry
            addLogEntry(`Prediction: ${prediction.state.toUpperCase()} (${(prediction.confidence * 100).toFixed(0)}% confidence)`);
        }
        
        // Update brainwave chart
        function updateBrainwave(state) {
            const sample = simulateEEGSample(state);
            
            brainwaveData.time.push(brainwaveData.time.length * 0.004);
            brainwaveData.signal.push(sample);
            
            if (brainwaveData.time.length > brainwaveData.maxPoints) {
                brainwaveData.time.shift();
                brainwaveData.signal.shift();
            }
            
            Plotly.restyle('brainwaveChart', {
                x: [brainwaveData.time],
                y: [brainwaveData.signal]
            });
        }
        
        // Update recommendations
        function updateRecommendations(prediction) {
            const container = document.getElementById('recommendationsContainer');
            
            let recommendations = [];
            let title = '';
            
            if (prediction.state === 'focused') {
                title = '✅ Great! You\'re in a focused state';
                recommendations = [
                    'Continue your current task - you\'re performing well',
                    'This is a good time for complex problem-solving',
                    'Maintain your current environment and routine'
                ];
            } else {
                const fatiguePercent = (fatiguedSeconds / (focusedSeconds + fatiguedSeconds)) * 100;
                
                if (fatiguePercent > 60) {
                    title = '🚨 High fatigue detected - Action recommended';
                    recommendations = [
                        'Take a 10-15 minute break immediately',
                        'Consider a power nap (15-20 minutes)',
                        'Get some fresh air or light exercise',
                        'Hydrate and have a healthy snack'
                    ];
                } else if (fatiguePercent > 40) {
                    title = '⚠️ Fatigue increasing - Plan a break';
                    recommendations = [
                        'Schedule a break in the next 15 minutes',
                        'Switch to less demanding tasks',
                        'Try the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec',
                        'Consider a brief stretching session'
                    ];
                } else {
                    title = '💡 Mild fatigue detected';
                    recommendations = [
                        'Continue working but monitor your state',
                        'Ensure good posture and lighting',
                        'Stay hydrated',
                        'Plan a break within 30 minutes'
                    ];
                }
            }
            
            let html = `<div class="recommendation-title">${title}</div><ul class="recommendation-list">`;
            recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            html += '</ul>';
            
            container.innerHTML = html;
        }
        
        // Show alert
        function showAlert(message, type = 'info') {
            const alertContainer = document.getElementById('alertContainer');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            alertDiv.style.display = 'block';
            
            alertContainer.innerHTML = '';
            alertContainer.appendChild(alertDiv);
            
            setTimeout(() => {
                alertDiv.style.display = 'none';
            }, 5000);
        }
        
        // Add log entry
        function addLogEntry(message) {
            const logContainer = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            const now = new Date();
            const timestamp = now.toTimeString().split(' ')[0];
            
            entry.innerHTML = `<span class="timestamp">[${timestamp}]</span><span>${message}</span>`;
            logContainer.insertBefore(entry, logContainer.firstChild);
            
            // Keep only last 20 entries
            while (logContainer.children.length > 20) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
        
        // Start monitoring
        function startMonitoring() {
            if (monitoringActive) return;
            
            monitoringActive = true;
            sessionStart = new Date();
            
            addLogEntry('Monitoring started');
            showAlert('Real-time monitoring active', 'info');
            
            // Update every second
            const monitorInterval = setInterval(() => {
                if (!monitoringActive) {
                    clearInterval(monitorInterval);
                    return;
                }
                
                const prediction = makePrediction();
                updateDashboard(prediction);
                updateBrainwave(prediction.state);
                
            }, 1000);
        }
        
        // Pause monitoring
        function pauseMonitoring() {
            monitoringActive = false;
            addLogEntry('Monitoring paused');
            showAlert('Monitoring paused', 'info');
        }
        
        // Reset monitoring
        function resetMonitoring() {
            monitoringActive = false;
            focusedSeconds = 0;
            fatiguedSeconds = 0;
            totalPredictions = 0;
            predictionHistory = [];
            brainwaveData = { time: [], signal: [], maxPoints: 100 };
            
            document.getElementById('statusBadge').textContent = 'READY';
            document.getElementById('statusBadge').className = 'status-badge status-unknown';
            
            initializeCharts();
            
            addLogEntry('System reset');
            showAlert('All data cleared. Ready to start new session.', 'info');
        }
        
        // Initialize on load
        window.onload = function() {
            initializeCharts();
            addLogEntry('Dashboard loaded successfully');
        };
    </script>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Interactive dashboard created: {output_path}")
    print("  Open this file in a web browser to see the real-time monitoring interface")
    
    return output_path


if __name__ == "__main__":
    create_dashboard_html()
