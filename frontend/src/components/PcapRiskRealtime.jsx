import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

const PcapRiskRealtime = () => {
  const [packets, setPackets] = useState([]);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    // Khởi tạo socket connection
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket', 'polling'], // Fallback to polling if websocket fails
      timeout: 20000,
      forceNew: true
    });

    const socket = socketRef.current;

    // Xử lý kết nối thành công
    socket.on('connect', () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
      setError(null);
      
      // Gọi API để bắt đầu xử lý PCAP
      fetch('http://localhost:5000/start-pcap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('PCAP started:', data.message);
      })
      .catch(err => {
        console.error('Lỗi khi gọi API:', err);
        setError(`API Error: ${err.message}`);
      });
    });

    // Xử lý ngắt kết nối
    socket.on('disconnect', (reason) => {
      console.log('Disconnected from server:', reason);
      setIsConnected(false);
    });

    // Xử lý lỗi kết nối
    socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setError(`Connection error: ${error.message}`);
      setIsConnected(false);
    });

    // Nhận dữ liệu gói tin
    socket.on('packet_data', (data) => {
      console.log('Received packet:', data);
      setPackets((prevPackets) => {
        const newPackets = [...prevPackets, data];
        return newPackets.slice(-100); // Giữ tối đa 100 gói tin
      });
    });

    // Nhận lỗi từ server
    socket.on('pcap_error', (data) => {
      console.error('PCAP Error:', data.message);
      setError(data.message);
    });

    // Cleanup khi component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, []);

  const handleReconnect = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current.connect();
    }
  };

  const clearError = () => {
    setError(null);
  };

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">Real-Time PCAP Risk Monitoring</h1>
        <div className="flex items-center gap-4">
          <div className={`px-3 py-1 rounded-full text-sm ${
            isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          {!isConnected && (
            <button
              onClick={handleReconnect}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Reconnect
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 flex justify-between items-center">
          <span>{error}</span>
          <button
            onClick={clearError}
            className="text-red-700 hover:text-red-900 font-bold"
          >
            ×
          </button>
        </div>
      )}

      <div className="mb-4 text-sm text-gray-600">
        Total packets received: {packets.length}
      </div>

      <div className="overflow-x-auto shadow-lg rounded-lg">
        <table className="min-w-full bg-white border border-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="py-3 px-4 border-b text-left font-medium text-gray-700">Source IP</th>
              <th className="py-3 px-4 border-b text-left font-medium text-gray-700">Destination IP</th>
              <th className="py-3 px-4 border-b text-left font-medium text-gray-700">Protocol</th>
              <th className="py-3 px-4 border-b text-left font-medium text-gray-700">Timestamp</th>
              <th className="py-3 px-4 border-b text-left font-medium text-gray-700">Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {packets.length === 0 ? (
              <tr>
                <td colSpan="5" className="py-8 px-4 text-center text-gray-500">
                  {isConnected ? 'Waiting for packet data...' : 'Not connected to server'}
                </td>
              </tr>
            ) : (
              packets.map((packet, index) => (
                <tr key={index} className="hover:bg-gray-50 transition-colors">
                  <td className="py-2 px-4 border-b text-sm">{packet.src_ip}</td>
                  <td className="py-2 px-4 border-b text-sm">{packet.dst_ip}</td>
                  <td className="py-2 px-4 border-b text-sm">{packet.protocol}</td>
                  <td className="py-2 px-4 border-b text-sm">
                    {new Date(packet.timestamp * 1000).toLocaleString()}
                  </td>
                  <td className={`py-2 px-4 border-b text-sm font-medium ${
                    packet.risk_level === 'high' 
                      ? 'text-red-600' 
                      : packet.risk_level === 'medium' 
                        ? 'text-yellow-600' 
                        : 'text-green-600'
                  }`}>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      packet.risk_level === 'high' 
                        ? 'bg-red-100' 
                        : packet.risk_level === 'medium' 
                          ? 'bg-yellow-100' 
                          : 'bg-green-100'
                    }`}>
                      {packet.risk_level}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PcapRiskRealtime;