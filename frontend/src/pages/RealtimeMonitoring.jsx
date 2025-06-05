import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { io } from 'socket.io-client';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { startRealtimeMonitoring, stopRealtimeMonitoring } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const RealtimeMonitoring = () => {
  const [socket, setSocket] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [riskData, setRiskData] = useState([]);
  const [currentStats, setCurrentStats] = useState({
    totalFlows: 0,
    highRiskFlows: 0,
    avgConfidence: 0,
    riskDistribution: {},
  });
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [lastUpdate, setLastUpdate] = useState(null);
  const fileInputRef = useRef(null);

  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: 'High Risk Flows',
        data: [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        tension: 0.1,
      },
      {
        label: 'Total Flows',
        data: [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        tension: 0.1,
      },
    ],
  });

  const getRiskDistributionData = useCallback(() => {
    const distribution = currentStats.riskDistribution;
    const labels = Object.keys(distribution);
    const data = Object.values(distribution);

    const colors = {
      'Không có': '#10b981',
      'Thấp': '#f59e0b',
      'Trung bình': '#ef4444',
      'Cao': '#dc2626',
      'Rất cao': '#7c2d12',
    };

    return {
      labels: labels,
      datasets: [
        {
          label: 'Risk Distribution',
          data: data,
          backgroundColor: labels.map((label) => colors[label] || '#6b7280'),
          borderWidth: 2,
          borderColor: '#ffffff',
        },
      ],
    };
  }, [currentStats.riskDistribution]);

  const lineChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: false },
    },
    scales: {
      y: { beginAtZero: true, title: { display: true, text: 'Number of Flows' } },
      x: { title: { display: true, text: 'Time' } },
    },
    animation: { duration: 0 },
  }), []);

  const doughnutChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'right' } },
  }), []);

  useEffect(() => {
    const newSocket = io('http://localhost:5000', {
      auth: { token: localStorage.getItem('token') },
      withCredentials: true,
    });

    newSocket.on('connect', () => {
      console.log('[Socket] Connected to server with SID:', newSocket.id);
    });

    newSocket.on('connect_error', (err) => {
      console.error('[Socket] Connection error:', err.message);
      setError(`Socket connection failed: ${err.message}`);
    });

    newSocket.on('join_success', (data) => {
      console.log('[Socket] Join success:', data);
    });

    newSocket.on('realtime_risk_update', (data) => {
      console.log('[Socket] Received realtime_risk_update:', data);
      if (data.session_id === sessionId) {
        const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
        setRiskData((prev) => [...prev.slice(-19), data]);

        setCurrentStats({
          totalFlows: data.risk_data.total_flows || 0,
          highRiskFlows: data.risk_data.high_risk_flows || 0,
          avgConfidence: data.risk_data.avg_confidence || 0,
          riskDistribution: data.risk_data.risk_distribution || {},
        });

        setChartData((prev) => ({
          ...prev,
          labels: [...prev.labels.slice(-19), timestamp],
          datasets: [
            {
              ...prev.datasets[0],
              data: [...prev.datasets[0].data.slice(-19), data.risk_data.high_risk_flows || 0],
            },
            {
              ...prev.datasets[1],
              data: [...prev.datasets[1].data.slice(-19), data.risk_data.total_flows || 0],
            },
          ],
        }));

        setLastUpdate(Date.now());
      } else {
        console.warn('[Socket] Session ID mismatch:', data.session_id, sessionId);
      }
    });

    newSocket.on('realtime_error', (data) => {
      console.error('[Socket] Received realtime_error:', data);
      if (data.session_id === sessionId) {
        setError(`Monitoring error: ${data.error}`);
        setIsMonitoring(false);
      }
    });

    setSocket(newSocket);

    return () => {
      console.log('[Socket] Disconnecting from server with SID:', newSocket.id);
      newSocket.close();
    };
  }, [sessionId]);

  useEffect(() => {
    if (!isMonitoring || !lastUpdate) return;

    const timeoutInterval = setInterval(() => {
      const timeSinceLastUpdate = Date.now() - lastUpdate;
      if (timeSinceLastUpdate > 10000) {
        console.warn('[Frontend] No data received for 10 seconds');
        setError('No data received for 10 seconds. Monitoring may have stopped.');
        setIsMonitoring(false);
      }
    }, 5000);

    return () => clearInterval(timeoutInterval);
  }, [isMonitoring, lastUpdate]);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    console.log('[Frontend] File selected:', file);
    if (!file || !file.name || typeof file.name !== 'string') {
      setError('Invalid file selected');
      setSelectedFile(null);
      console.error('[Frontend] Invalid file or file name:', file);
      return;
    }

    if (file.name.toLowerCase().endsWith('.pcap')) {
      setSelectedFile(file);
      setError('');
      console.log('[Frontend] Selected file:', file.name);
    } else {
      setError('Please select a valid PCAP file');
      setSelectedFile(null);
      console.error('[Frontend] Invalid file extension:', file.name);
    }
  };

  const startMonitoring = async () => {
    if (!selectedFile) {
      setError('Please select a PCAP file');
      console.error('[Frontend] No file selected for monitoring');
      return;
    }

    setIsUploading(true);
    setError('');
    console.log('[Frontend] Starting monitoring with file:', selectedFile.name);

    try {
      const data = await startRealtimeMonitoring(selectedFile);
      console.log('[Frontend] Start monitoring response:', data);

      if (!data.error) {
        setSessionId(data.session_id);
        setIsMonitoring(true);
        setRiskData([]);
        setCurrentStats({
          totalFlows: 0,
          highRiskFlows: 0,
          avgConfidence: 0,
          riskDistribution: {},
        });
        setLastUpdate(null);

        if (socket) {
          socket.emit('join', data.session_id);
          console.log('[Socket] Joined session:', data.session_id);
        }
      } else {
        setError(data.error || 'Failed to start monitoring');
        console.error('[Frontend] Failed to start monitoring:', data.error);
      }
    } catch (err) {
      setError('Network error occurred');
      console.error('[Frontend] Network error during start monitoring:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const stopMonitoring = async () => {
    if (!sessionId) return;

    console.log('[Frontend] Stopping monitoring for session:', sessionId);
    try {
      const response = await stopRealtimeMonitoring(sessionId);
      console.log('[Frontend] Stop monitoring response:', response);

      if (!response.error) {
        setIsMonitoring(false);
        setSessionId(null);
        if (socket) {
          socket.emit('leave', sessionId);
          console.log('[Socket] Left session:', sessionId);
        }
      } else {
        console.error('[Frontend] Error stopping monitoring:', response.error);
      }
    } catch (err) {
      console.error('[Frontend] Error stopping monitoring:', err);
    }
  };

  const getRiskLevel = (highRiskFlows, totalFlows) => {
    if (totalFlows === 0) return 'Unknown';
    const ratio = highRiskFlows / totalFlows;
    if (ratio >= 0.7) return 'Critical';
    if (ratio >= 0.4) return 'High';
    if (ratio >= 0.2) return 'Medium';
    if (ratio > 0) return 'Low';
    return 'Safe';
  };

  const getRiskLevelColor = (level) => {
    const colors = {
      'Critical': 'text-red-800 bg-red-100',
      'High': 'text-red-700 bg-red-50',
      'Medium': 'text-yellow-700 bg-yellow-50',
      'Low': 'text-blue-700 bg-blue-50',
      'Safe': 'text-green-700 bg-green-50',
      'Unknown': 'text-gray-700 bg-gray-50',
    };
    return colors[level] || colors['Unknown'];
  };

  const overallRiskLevel = getRiskLevel(currentStats.highRiskFlows, currentStats.totalFlows);
  const riskLevelColor = getRiskLevelColor(overallRiskLevel);

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Realtime APT Risk Monitoring
          </h3>

          <div className="mb-6">
            <div className="flex items-center space-x-4">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pcap"
                onChange={handleFileSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isMonitoring}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              >
                Select PCAP File
              </button>

              {selectedFile && (
                <span className="text-sm text-gray-600">
                  Selected: {selectedFile.name}
                </span>
              )}
            </div>
          </div>

          <div className="flex space-x-4 mb-6">
            {!isMonitoring ? (
              <button
                onClick={startMonitoring}
                disabled={!selectedFile || isUploading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              >
                {isUploading ? 'Starting...' : 'Start Monitoring'}
              </button>
            ) : (
              <button
                onClick={stopMonitoring}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                Stop Monitoring
              </button>
            )}
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          <div className="mb-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
              <span className="text-sm font-medium text-gray-700">
                {isMonitoring ? 'Monitoring Active' : 'Monitoring Inactive'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {isMonitoring && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                    <span className="text-white text-sm font-medium">T</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Total Flows
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {currentStats.totalFlows}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-red-500 rounded-md flex items-center justify-center">
                    <span className="text-white text-sm font-medium">H</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      High Risk Flows
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {currentStats.highRiskFlows}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-green-500 rounded-md flex items-center justify-center">
                    <span className="text-white text-sm font-medium">C</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Avg Confidence
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {(currentStats.avgConfidence * 100).toFixed(1)}%
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 rounded-md flex items-center justify-center ${
                    overallRiskLevel === 'Critical' ? 'bg-red-600' :
                    overallRiskLevel === 'High' ? 'bg-red-500' :
                    overallRiskLevel === 'Medium' ? 'bg-yellow-500' :
                    overallRiskLevel === 'Low' ? 'bg-blue-500' : 'bg-green-500'
                  }`}>
                    <span className="text-white text-sm font-medium">R</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      Risk Level
                    </dt>
                    <dd className={`text-lg font-medium px-2 py-1 rounded-full text-center ${riskLevelColor}`}>
                      {overallRiskLevel}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {isMonitoring && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white shadow rounded-lg p-6">
            <h4 className="text-lg font-medium text-gray-900 mb-4">Risk Trends Over Time</h4>
            <div className="h-64">
              {chartData.labels.length > 0 ? (
                <Line
                  data={chartData}
                  options={lineChartOptions}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  Waiting for data...
                </div>
              )}
            </div>
          </div>

          <div className="bg-white shadow rounded-lg p-6">
            <h4 className="text-lg font-medium text-gray-900 mb-4">Current Risk Distribution</h4>
            <div className="h-64">
              {Object.keys(currentStats.riskDistribution).length > 0 ? (
                <Doughnut
                  data={getRiskDistributionData()}
                  options={doughnutChartOptions}
                />
              ) : (
                <div className="flex items-center h-5 text-gray-200">
                  No data available
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RealtimeMonitoring;