import React, { useState, useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';
import { simulate } from '../services/api';
import { AlertCircle, Loader2 } from 'lucide-react';

const Simulation = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const eventSourceRef = useRef(null);

  const phaseColors = {
    Benign: 'rgba(75, 192, 192, 1)',
    Reconnaissance: 'rgba(255, 99, 132, 1)',
    'Establish Foothold': 'rgba(255, 206, 86, 1)',
    'Lateral Movement': 'rgba(54, 162, 235, 1)',
    'Data Exfiltration': 'rgba(153, 102, 255, 1)',
  };

  const stages = ['Benign', 'Reconnaissance', 'Establish Foothold', 'Lateral Movement', 'Data Exfiltration'];

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    if (chartRef.current) {
      const ctx = chartRef.current.getContext('2d');
      chartInstanceRef.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: '',
              data: [],
              backgroundColor: [],
              borderColor: 'rgba(100, 100, 100, 0.3)',
              pointBackgroundColor: [],
              pointBorderColor: [],
              pointRadius: 6,
              pointHoverRadius: 8,
              borderWidth: 2,
              fill: false,
              tension: 0,
              pointBorderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            intersect: true,
            mode: 'point'
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Log index',
                font: { size: 14, weight: 'bold' },
                color: '#1f2937',
              },
              grid: { display: false },
              ticks: {
                display: true,
                font: { size: 12 },
                color: '#4b5563',
              }
            },
            y: {
              title: {
                display: true,
                text: 'Giai đoạn tấn công',
                font: { size: 14, weight: 'bold' },
                color: '#1f2937',
              },
              min: 0,
              max: 4,
              ticks: {
                stepSize: 1,
                font: { size: 12 },
                color: '#4b5563',
                callback: function(value) {
                  return stages[value] || '';
                }
              },
              grid: { color: '#e5e7eb' },
            },
          },
          plugins: {
            tooltip: {
              enabled: true,
              external: null,
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              titleColor: 'white',
              bodyColor: 'white',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              borderWidth: 1,
              titleFont: { size: 14 },
              bodyFont: { size: 12 },
              displayColors: false,
              padding: 10,
              cornerRadius: 6,
              caretPadding: 4,
              callbacks: {
                title: function() {
                  return '';
                },
                label: function(context) {
                  const index = context.dataIndex;
                  const logIndex = context.label;
                  const stageValue = context.parsed.y;
                  const stageName = stages[stageValue] || 'Không xác định';
                  return `Log ${logIndex}: ${stageName}`;
                },
              },
            },
            legend: {
              display: false
            },
            datalabels: {
              display: false
            }
          },
          elements: {
            point: {
              hoverRadius: 8,
              radius: 6,
            },
            line: {
              borderWidth: 2,
            }
          },
          layout: {
            padding: 0
          },
          animation: {
            duration: 0
          }
        },
      });
    }

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  const resetChart = () => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.data.labels = [];
      chartInstanceRef.current.data.datasets[0].data = [];
      chartInstanceRef.current.data.datasets[0].backgroundColor = [];
      chartInstanceRef.current.data.datasets[0].pointBackgroundColor = [];
      chartInstanceRef.current.data.datasets[0].pointBorderColor = [];
      chartInstanceRef.current.update('none');
    }
  };

  const updateChart = (index, prediction) => {
    if (chartInstanceRef.current) {
      const stageIndex = stages.indexOf(prediction.stage_label);
      const finalStageIndex = stageIndex !== -1 ? stageIndex : 0;
      const color = phaseColors[prediction.stage_label] || phaseColors.Benign;
      
      chartInstanceRef.current.data.labels.push(index);
      chartInstanceRef.current.data.datasets[0].data.push(finalStageIndex);
      chartInstanceRef.current.data.datasets[0].pointBackgroundColor.push(color);
      chartInstanceRef.current.data.datasets[0].pointBorderColor.push(color);
      chartInstanceRef.current.update('none');
    }
  };

  const handleFileChange = (e) => {
    const newFile = e.target.files[0];
    if (!newFile) return;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setFile(newFile);
    setPredictions([]);
    setError('');
    resetChart();
  };

  const processFile = () => {
    if (!file) {
      setError('Vui lòng chọn một tệp');
      return;
    }

    setIsProcessing(true);
    setError('');

    simulate(file, (data) => {
      if (data.error) {
        setError(data.error || 'Đã có lỗi trong quá trình xử lý');
        setIsProcessing(false);
      } else {
        setPredictions(prev => [...prev, data]);
        updateChart(data.log_index + 1, data);
      }
    }, (err) => {
      setError(err?.toString() || 'Lỗi không xác định');
      setIsProcessing(false);
    }).then(eventSource => {
      eventSourceRef.current = eventSource;
    }).catch(err => {
      setError(err?.toString() || 'Không thể bắt đầu mô phỏng');
      setIsProcessing(false);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
             Mô phỏng phân loại giai đoạn tấn công APT thời gian thực 
          </h1>
          <p className="text-gray-600 mt-2">Phân tích các tệp log để xác định các giai đoạn tấn công APT</p>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
          <div className="p-4 sm:p-6">
            <div className="flex items-center space-x-4 mb-6">
              <input
                type="file"
                accept=".csv,.pcap"
                onChange={handleFileChange}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <button
                onClick={processFile}
                disabled={isProcessing}
                className={`px-6 py-2 rounded-md text-white font-semibold transition-colors duration-200 ${
                  isProcessing ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isProcessing ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Đang xử lý...
                  </div>
                ) : (
                  'Bắt đầu mô phỏng'
                )}
              </button>
            </div>

            <div className="mb-6">
              <div className="relative h-96">
                <canvas ref={chartRef}></canvas>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">Nhật ký dự đoán</h2>
              {predictions.length === 0 && !isProcessing ? (
                <div className="flex flex-col items-center gap-4 py-8">
                  <p className="text-gray-600 text-lg font-medium">Chưa có dự đoán nào</p>
                  <p className="text-gray-500">Hãy upload tệp log để bắt đầu phân tích.</p>
                </div>
              ) : (
                <ul className="max-h-60 overflow-y-auto space-y-2">
                  {predictions.map((pred, idx) => (
                    <li
                      key={idx}
                      className="flex items-center space-x-2 p-2 bg-white rounded-md border{(index % 2 === 0 ? 'bg-gray-50' : 'bg-white')} border-gray-200"
                    >
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: phaseColors[pred.stage_label] || phaseColors.Benign }}
                      ></span>
                      <span className="font-medium text-gray-700">Log {pred.log_index + 1}:</span>
                      <span className="text-gray-600">{pred.stage_label}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Simulation;