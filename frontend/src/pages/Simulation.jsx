// import React, { useState, useEffect, useRef } from "react";
// import Chart from "chart.js/auto";
// import { simulate, getQueueStatus } from "../services/api";
// import { AlertCircle, Loader2, Upload, FileText, Clock } from "lucide-react";

// const Simulation = () => {
//   const [predictions, setPredictions] = useState([]);
//   const [isProcessing, setIsProcessing] = useState(false);
//   const [error, setError] = useState("");
//   const [queueStatus, setQueueStatus] = useState({
//     queue_length: 0,
//     is_processing: false,
//     queue_files: [],
//   });
//   const [queuedFiles, setQueuedFiles] = useState([]); // Track pending files
//   const [currentFile, setCurrentFile] = useState("");
//   const [totalProcessed, setTotalProcessed] = useState(0);
//   const [statusMessage, setStatusMessage] = useState("");

//   const chartRef = useRef(null);
//   const chartInstanceRef = useRef(null);
//   const eventSourceRef = useRef(null);
//   const fileInputRef = useRef(null);

//   const phaseColors = {
//     Benign: "rgba(75, 192, 192, 1)",
//     Reconnaissance: "rgba(255, 99, 132, 1)",
//     "Establish Foothold": "rgba(255, 206, 86, 1)",
//     "Lateral Movement": "rgba(54, 162, 235, 1)",
//     "Data Exfiltration": "rgba(153, 102, 255, 1)",
//   };

//   const stages = [
//     "Benign",
//     "Reconnaissance",
//     "Establish Foothold",
//     "Lateral Movement",
//     "Data Exfiltration",
//   ];

//   // Auto refresh queue status
//   useEffect(() => {
//     const interval = setInterval(async () => {
//       try {
//         const status = await getQueueStatus();
//         console.log("[Simulation] Queue status update:", status);
//         setQueueStatus(status);
//       } catch (err) {
//         console.error("[Simulation] Failed to fetch queue status:", err);
//       }
//     }, 2000);

//     return () => clearInterval(interval);
//   }, []);

//   useEffect(() => {
//     if (error) {
//       const timer = setTimeout(() => setError(""), 5000);
//       return () => clearTimeout(timer);
//     }
//   }, [error]);

//   useEffect(() => {
//     if (statusMessage) {
//       const timer = setTimeout(() => setStatusMessage(""), 3000);
//       return () => clearTimeout(timer);
//     }
//   }, [statusMessage]);

//   useEffect(() => {
//     if (chartRef.current) {
//       const ctx = chartRef.current.getContext("2d");
//       chartInstanceRef.current = new Chart(ctx, {
//         type: "line",
//         data: {
//           labels: [],
//           datasets: [
//             {
//               label: "",
//               data: [],
//               backgroundColor: [],
//               borderColor: "rgba(100, 100, 100, 0.3)",
//               pointBackgroundColor: [],
//               pointBorderColor: [],
//               pointRadius: 6,
//               pointHoverRadius: 8,
//               borderWidth: 2,
//               fill: false,
//               tension: 0,
//               pointBorderWidth: 2,
//             },
//           ],
//         },
//         options: {
//           responsive: true,
//           maintainAspectRatio: false,
//           interaction: {
//             intersect: true,
//             mode: "point",
//           },
//           scales: {
//             x: {
//               title: {
//                 display: true,
//                 text: "Log index (liên tục)",
//                 font: { size: 14, weight: "bold" },
//                 color: "#1f2937",
//               },
//               grid: { display: false },
//               ticks: {
//                 display: true,
//                 font: { size: 12 },
//                 color: "#4b5563",
//               },
//             },
//             y: {
//               title: {
//                 display: true,
//                 text: "Giai đoạn tấn công",
//                 font: { size: 14, weight: "bold" },
//                 color: "#1f2937",
//               },
//               min: 0,
//               max: 4,
//               ticks: {
//                 stepSize: 1,
//                 font: { size: 12 },
//                 color: "#4b5563",
//                 callback: function (value) {
//                   return stages[value] || "";
//                 },
//               },
//               grid: { color: "#e5e7eb" },
//             },
//           },
//           plugins: {
//             tooltip: {
//               enabled: true,
//               backgroundColor: "rgba(0, 0, 0, 0.8)",
//               titleColor: "white",
//               bodyColor: "white",
//               borderColor: "rgba(255, 255, 255, 0.2)",
//               borderWidth: 1,
//               titleFont: { size: 14 },
//               bodyFont: { size: 12 },
//               displayColors: false,
//               padding: 10,
//               cornerRadius: 6,
//               caretPadding: 4,
//               callbacks: {
//                 title: function () {
//                   return "";
//                 },
//                 label: function (context) {
//                   const index = context.dataIndex;
//                   const logIndex = context.label;
//                   const stageValue = context.parsed.y;
//                   const stageName = stages[stageValue] || "Không xác định";
//                   const filename = predictions[index]?.filename || "Unknown";
//                   const fileLogIndex = predictions[index]?.file_log_index || 0;
//                   return [
//                     `Log tổng: ${logIndex}`,
//                     `File: ${filename}`,
//                     `Log trong file: ${fileLogIndex + 1}`,
//                     `Giai đoạn: ${stageName}`,
//                   ];
//                 },
//               },
//             },
//             legend: {
//               display: false,
//             },
//             datalabels: {
//               display: false,
//             },
//           },
//           elements: {
//             point: {
//               hoverRadius: 8,
//               radius: 6,
//             },
//             line: {
//               borderWidth: 2,
//             },
//           },
//           layout: {
//             padding: 0,
//           },
//           animation: {
//             duration: 300,
//           },
//         },
//       });
//     }

//     return () => {
//       if (chartInstanceRef.current) {
//         chartInstanceRef.current.destroy();
//         chartInstanceRef.current = null;
//       }
//       if (eventSourceRef.current) {
//         eventSourceRef.current.close();
//         eventSourceRef.current = null;
//       }
//     };
//   }, []);

//   const resetSimulation = () => {
//     if (chartInstanceRef.current) {
//       chartInstanceRef.current.data.labels = [];
//       chartInstanceRef.current.data.datasets[0].data = [];
//       chartInstanceRef.current.data.datasets[0].backgroundColor = [];
//       chartInstanceRef.current.data.datasets[0].pointBackgroundColor = [];
//       chartInstanceRef.current.data.datasets[0].pointBorderColor = [];
//       chartInstanceRef.current.update("none");
//     }
//     setPredictions([]);
//     setTotalProcessed(0);
//     setCurrentFile("");
//     setStatusMessage("");
//     setQueuedFiles([]); // Clear queued files on reset
//   };

//   const updateChart = (prediction) => {
//     if (chartInstanceRef.current) {
//       const stageIndex = stages.indexOf(prediction.stage_label);
//       const finalStageIndex = stageIndex !== -1 ? stageIndex : 0;
//       const color = phaseColors[prediction.stage_label] || phaseColors.Benign;

//       chartInstanceRef.current.data.labels.push(prediction.log_index + 1);
//       chartInstanceRef.current.data.datasets[0].data.push(finalStageIndex);
//       chartInstanceRef.current.data.datasets[0].pointBackgroundColor.push(
//         color
//       );
//       chartInstanceRef.current.data.datasets[0].pointBorderColor.push(color);
//       chartInstanceRef.current.update("none");
//     }
//   };

//   const handleFileChange = async (e) => {
//     const selectedFiles = Array.from(e.target.files).filter(
//       (file) =>
//         file.name.toLowerCase().endsWith(".csv") ||
//         file.name.toLowerCase().endsWith(".pcap")
//     );

//     if (selectedFiles.length === 0) {
//       setError("Vui lòng chọn tệp CSV hoặc PCAP");
//       return;
//     }

//     setError("");

//     // Upload each file sequentially
//     for (const file of selectedFiles) {
//       try {
//         console.log("[Simulation] Uploading file:", file.name);
//         await uploadAndProcessFile(file);
//       } catch (err) {
//         console.error(`[Simulation] Failed to upload ${file.name}:`, err);
//         setError(`Không thể tải lên ${file.name}: ${err.message || err}`);
//         break;
//       }
//     }

//     // Reset file input
//     if (fileInputRef.current) {
//       fileInputRef.current.value = "";
//     }
//   };

//   const uploadAndProcessFile = (file) => {
//     return new Promise((resolve, reject) => {
//       const eventSource = simulate(
//         file,
//         (data) => {
//           console.log("[Simulation] Received data for file:", file.name, data);
//           if (data.error) {
//             setError(data.error);
//             setIsProcessing(false);
//             reject(new Error(data.error));
//           } else if (data.status) {
//             handleStatusUpdate(data);
//           } else {
//             // Regular prediction data
//             setPredictions((prev) => [
//               ...prev,
//               {
//                 ...data,
//                 filename: data.filename,
//                 file_log_index: data.file_log_index || 0,
//               },
//             ]);
//             updateChart(data);
//             setTotalProcessed(data.log_index + 1);
//           }
//         },
//         (err) => {
//           console.error("[Simulation] Error processing file:", file.name, err);
//           setError(err?.toString() || "Lỗi không xác định");
//           setIsProcessing(false);
//           reject(err);
//         }
//       );

//       eventSourceRef.current = eventSource;
//       resolve();
//     });
//   };

//   const handleStatusUpdate = (data) => {
//     console.log("[Simulation] Status update:", data);
//     switch (data.status) {
//       case "file_queued":
//         setStatusMessage(`Đã thêm file ${data.filename} vào hàng đợi`);
//         setQueueStatus((prev) => ({
//           ...prev,
//           queue_length: data.queue_length,
//         }));
//         setQueuedFiles((prev) => {
//           const updated = [...prev, data.filename];
//           console.log(
//             "[Simulation] Updated queuedFiles (file_queued):",
//             updated
//           );
//           return updated;
//         });
//         break;
//       case "file_started":
//         setCurrentFile(data.filename);
//         setStatusMessage(`Bắt đầu xử lý file: ${data.filename}`);
//         setIsProcessing(true);
//         break;
//       case "file_completed":
//         setStatusMessage(
//           `Hoàn thành xử lý file: ${data.filename} (${data.total_logs} logs)`
//         );
//         setQueuedFiles((prev) => {
//           const updated = prev.filter((file) => file !== data.filename);
//           console.log(
//             "[Simulation] Updated queuedFiles (file_completed):",
//             updated
//           );
//           return updated;
//         });
//         break;
//       case "all_completed":
//         setStatusMessage(
//           `Hoàn thành tất cả! Tổng cộng đã xử lý ${data.total_logs_processed} logs`
//         );
//         setIsProcessing(false);
//         setCurrentFile("");
//         setQueuedFiles([]); // Clear all queued files
//         console.log("[Simulation] Cleared queuedFiles (all_completed)");
//         break;
//       default:
//         break;
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
//       <div className="container mx-auto px-4 py-12 max-w-6xl">
//         {/* Header */}
//         <div className="text-center mb-10">
//           <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
//             Mô phỏng phân loại giai đoạn tấn công APT thời gian thực
//           </h1>
//           <p className="text-gray-600 mt-2">
//             Phân tích liên tục các tệp log để xác định các giai đoạn tấn công
//             APT
//           </p>
//         </div>

//         {/* Status Messages */}
//         {error && (
//           <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
//             <AlertCircle className="w-5 h-5" />
//             {error}
//           </div>
//         )}

//         {statusMessage && (
//           <div className="fixed top-4 left-4 max-w-sm p-4 bg-blue-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
//             <FileText className="w-5 h-5" />
//             {statusMessage}
//           </div>
//         )}

//         {/* Queue Status Card */}
//         {(queueStatus.queue_length > 0 ||
//           isProcessing ||
//           queuedFiles.length > 0) && (
//           <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-lg border border-gray-100 p-6 mb-8 transform transition-all hover:shadow-xl">
//             <div className="flex items-center justify-between">
//               <div className="flex items-center gap-4">
//                 <Clock className="w-6 h-6 text-blue-600 flex-shrink-0" />
//                 <div>
//                   <h3 className="text-lg font-bold text-gray-900 tracking-tight">
//                     Trạng thái xử lý
//                   </h3>
//                   <p className="text-sm font-medium text-gray-500 mt-1">
//                     {isProcessing ? `Đang xử lý: ${currentFile}` : "Đang chờ"}
//                   </p>
//                 </div>
//               </div>
//             </div>
//             {queuedFiles.length > 0 && (
//               <div className="mt-4 pt-4 border-t border-gray-200">
//                 <p className="text-sm font-semibold text-gray-700 mb-3">
//                   Files đang chờ:
//                 </p>
//                 <div className="flex flex-wrap gap-2">
//                   {queuedFiles.map((filename, idx) => (
//                     <span
//                       key={idx}
//                       className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium hover:bg-blue-100 transition-colors duration-200 truncate max-w-xs"
//                     >
//                       {filename}
//                     </span>
//                   ))}
//                 </div>
//               </div>
//             )}
//           </div>
//         )}

//         {/* Main Card */}
//         <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
//           <div className="p-4 sm:p-6">
//             <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-4 mb-6">
//               <div className="flex-1 min-w-0">
//                 <input
//                   type="file"
//                   accept=".csv,.pcap"
//                   multiple
//                   onChange={handleFileChange}
//                   ref={fileInputRef}
//                   className="w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 file:cursor-pointer cursor-pointer"
//                 />
//                 <p className="text-xs text-gray-500 mt-1">
//                   Có thể chọn nhiều file cùng lúc. Các file sẽ được xử lý tuần
//                   tự và liên tục.
//                 </p>
//               </div>
//               <div className="flex gap-2">
//                 <button
//                   onClick={resetSimulation}
//                   className="px-4 py-2 rounded-md text-gray-600 border border-gray-300 hover:bg-gray-50 font-semibold transition-colors duration-200 whitespace-nowrap"
//                 >
//                   Reset
//                 </button>
//               </div>
//             </div>

//             {/* Stats Row */}
//             {totalProcessed > 0 && (
//               <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
//                 <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300">
//                   <div className="text-3xl font-extrabold text-blue-700 tracking-tight">
//                     {totalProcessed}
//                   </div>
//                   <div className="text-sm font-medium text-blue-900 mt-1">
//                     Tổng logs đã xử lý
//                   </div>
//                 </div>
//                 <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300">
//                   <div className="text-3xl font-extrabold text-purple-700 tracking-tight">
//                     {isProcessing ? (
//                       <Loader2 className="w-7 h-7 animate-spin text-purple-600" />
//                     ) : (
//                       <span className="text-green-600">✓</span>
//                     )}
//                   </div>
//                   <div className="text-sm font-medium text-purple-900 mt-1">
//                     {isProcessing ? "Đang xử lý" : "Hoàn thành"}
//                   </div>
//                 </div>
//               </div>
//             )}

//             <div className="mb-6">
//               <div className="relative h-96">
//                 <canvas ref={chartRef}></canvas>
//               </div>
//             </div>

//             <div className="bg-gray-50 p-4 rounded-lg shadow">
//               <h2 className="text-xl font-semibold text-gray-800 mb-3">
//                 Nhật ký dự đoán
//               </h2>
//               {predictions.length === 0 && !isProcessing ? (
//                 <div className="flex flex-col items-center gap-4 py-8">
//                   <Upload className="w-12 h-12 text-gray-400" />
//                   <p className="text-gray-600 text-lg font-medium">
//                     Chưa có dự đoán nào
//                   </p>
//                   <p className="text-gray-500 text-center">
//                     Hãy upload các tệp log để bắt đầu phân tích liên tục.
//                     <br />
//                   </p>
//                 </div>
//               ) : (
//                 <ul className="max-h-96 overflow-y-auto space-y-2 p-2 rounded-lg bg-gray-50 border border-gray-100">
//                   {predictions.map((pred, idx) => (
//                     <li
//                       key={pred.log_index}
//                       className="flex items-center space-x-3 p-3 bg-white rounded-lg border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors duration-200"
//                     >
//                       <span
//                         className="w-3 h-3 rounded-full flex-shrink-0"
//                         style={{
//                           backgroundColor:
//                             phaseColors[pred.stage_label] || phaseColors.Benign,
//                         }}
//                       ></span>
//                       <div className="flex-1 min-w-0">
//                         <p className="text-sm font-semibold text-gray-800 truncate">
//                           Log {pred.log_index + 1}: {pred.stage_label}
//                         </p>
//                         <p className="text-xs text-gray-500 truncate">
//                           File: {pred.filename} (Log trong file:{" "}
//                           {pred.file_log_index + 1})
//                         </p>
//                       </div>
//                     </li>
//                   ))}
//                 </ul>
//               )}
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Simulation;

import React, { useState, useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { simulate, getQueueStatus } from "../services/api";
import { AlertCircle, Loader2, Upload, FileText, Clock } from "lucide-react";
import "chartjs-adapter-date-fns";

const Simulation = () => {
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [queueStatus, setQueueStatus] = useState({
    queue_length: 0,
    is_processing: false,
    queue_files: [],
  });
  const [queuedFiles, setQueuedFiles] = useState([]); // Track pending files
  const [currentFile, setCurrentFile] = useState("");
  const [totalProcessed, setTotalProcessed] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");

  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const eventSourceRef = useRef(null);
  const fileInputRef = useRef(null);

  const phaseColors = {
    Benign: "#6b7280",
    Reconnaissance: "#84cc16",
    "Establish Foothold": "#facc15",
    "Lateral Movement": "#f97316",
    "Data Exfiltration": "#dc2626",
  };

  const stages = [
    "Benign",
    "Reconnaissance",
    "Establish Foothold",
    "Lateral Movement",
    "Data Exfiltration",
  ];

  // Auto refresh queue status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const status = await getQueueStatus();
        console.log("[Simulation] Queue status update:", status);
        setQueueStatus(status);
      } catch (err) {
        console.error("[Simulation] Failed to fetch queue status:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(""), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    if (statusMessage) {
      const timer = setTimeout(() => setStatusMessage(""), 3000);
      return () => clearTimeout(timer);
    }
  }, [statusMessage]);

  useEffect(() => {
    if (chartRef.current) {
      const ctx = chartRef.current.getContext("2d");
      chartInstanceRef.current = new Chart(ctx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "",
              data: [],
              backgroundColor: [],
              borderColor: "rgba(100, 100, 100, 0.3)",
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
            mode: "point",
          },
          scales: {
            x: {
              type: "time",
              time: {
                unit: "second",
                displayFormats: {
                  second: "HH:mm:ss",
                },
                tooltipFormat: "HH:mm:ss",
              },
              title: {
                display: true,
                text: "Thời gian",
                font: { size: 14, weight: "bold" },
                color: "#1f2937",
              },
              grid: { display: false },
              ticks: {
                display: true,
                font: { size: 12 },
                color: "#4b5563",
                maxTicksLimit: 10,
              },
            },
            y: {
              title: {
                display: true,
                text: "Giai đoạn tấn công",
                font: { size: 14, weight: "bold" },
                color: "#1f2937",
              },
              min: 0,
              max: 4,
              ticks: {
                stepSize: 1,
                font: { size: 12 },
                color: "#4b5563",
                callback: function (value) {
                  return stages[value] || "";
                },
              },
              grid: { color: "#e5e7eb" },
            },
          },
          plugins: {
            tooltip: {
              enabled: true,
              backgroundColor: "rgba(0, 0, 0, 0.8)",
              titleColor: "white",
              bodyColor: "white",
              borderColor: "rgba(255, 255, 255, 0.2)",
              borderWidth: 1,
              titleFont: { size: 14 },
              bodyFont: { size: 12 },
              displayColors: false,
              padding: 10,
              cornerRadius: 6,
              caretPadding: 4,
              callbacks: {
                title: function () {
                  return "";
                },
                label: function (context) {
                  const index = context.dataIndex;
                  const timestamp = new Date(context.label).toLocaleTimeString("vi-VN", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  });
                  const stageValue = context.parsed.y;
                  const stageName = stages[stageValue] || "Không xác định";
                  const filename = predictions[index]?.filename || "Unknown";
                  const fileLogIndex = predictions[index]?.file_log_index || 0;
                  const logIndex = predictions[index]?.log_index || 0;
                  return [
                    `Thời gian: ${timestamp}`,
                    `Log tổng: ${logIndex + 1}`,
                    `File: ${filename}`,
                    `Log trong file: ${fileLogIndex + 1}`,
                    `Giai đoạn: ${stageName}`,
                  ];
                },
              },
            },
            legend: {
              display: false,
            },
            datalabels: {
              display: false,
            },
          },
          elements: {
            point: {
              hoverRadius: 8,
              radius: 6,
            },
            line: {
              borderWidth: 2,
            },
          },
          layout: {
            padding: 0,
          },
          animation: {
            duration: 300,
          },
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

  const resetSimulation = () => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.data.labels = [];
      chartInstanceRef.current.data.datasets[0].data = [];
      chartInstanceRef.current.data.datasets[0].backgroundColor = [];
      chartInstanceRef.current.data.datasets[0].pointBackgroundColor = [];
      chartInstanceRef.current.data.datasets[0].pointBorderColor = [];
      chartInstanceRef.current.update("none");
    }
    setPredictions([]);
    setTotalProcessed(0);
    setCurrentFile("");
    setStatusMessage("");
    setQueuedFiles([]); // Clear queued files on reset
  };

  const updateChart = (prediction) => {
    if (chartInstanceRef.current) {
      const stageIndex = stages.indexOf(prediction.stage_label);
      const finalStageIndex = stageIndex !== -1 ? stageIndex : 0;
      const color = phaseColors[prediction.stage_label] || phaseColors.Benign;

      chartInstanceRef.current.data.labels.push(prediction.timestamp);
      chartInstanceRef.current.data.datasets[0].data.push(finalStageIndex);
      chartInstanceRef.current.data.datasets[0].pointBackgroundColor.push(color);
      chartInstanceRef.current.data.datasets[0].pointBorderColor.push(color);
      chartInstanceRef.current.update("none");
    }
  };

  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files).filter(
      (file) =>
        file.name.toLowerCase().endsWith(".csv") ||
        file.name.toLowerCase().endsWith(".pcap")
    );

    if (selectedFiles.length === 0) {
      setError("Vui lòng chọn tệp CSV hoặc PCAP");
      return;
    }

    setError("");

    // Upload each file sequentially
    for (const file of selectedFiles) {
      try {
        console.log("[Simulation] Uploading file:", file.name);
        await uploadAndProcessFile(file);
      } catch (err) {
        console.error(`[Simulation] Failed to upload ${file.name}:`, err);
        setError(`Không thể tải lên ${file.name}: ${err.message || err}`);
        break;
      }
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const uploadAndProcessFile = (file) => {
    return new Promise((resolve, reject) => {
      const eventSource = simulate(
        file,
        (data) => {
          console.log("[Simulation] Received data for file:", file.name, data);
          if (data.error) {
            setError(data.error);
            setIsProcessing(false);
            reject(new Error(data.error));
          } else if (data.status) {
            handleStatusUpdate(data);
          } else {
            // Regular prediction data
            const predictionWithTimestamp = {
              ...data,
              filename: data.filename,
              file_log_index: data.file_log_index || 0,
              timestamp: new Date().toISOString(),
            };
            setPredictions((prev) => [...prev, predictionWithTimestamp]);
            updateChart(predictionWithTimestamp);
            setTotalProcessed(data.log_index + 1);
          }
        },
        (err) => {
          console.error("[Simulation] Error processing file:", file.name, err);
          setError(err?.toString() || "Lỗi không xác định");
          setIsProcessing(false);
          reject(err);
        }
      );

      eventSourceRef.current = eventSource;
      resolve();
    });
  };

  const handleStatusUpdate = (data) => {
    console.log("[Simulation] Status update:", data);
    switch (data.status) {
      case "file_queued":
        setStatusMessage(`Đã thêm file ${data.filename} vào hàng đợi`);
        setQueueStatus((prev) => ({
          ...prev,
          queue_length: data.queue_length,
        }));
        setQueuedFiles((prev) => {
          const updated = [...prev, data.filename];
          console.log(
            "[Simulation] Updated queuedFiles (file_queued):",
            updated
          );
          return updated;
        });
        break;
      case "file_started":
        setCurrentFile(data.filename);
        setStatusMessage(`Bắt đầu xử lý file: ${data.filename}`);
        setIsProcessing(true);
        break;
      case "file_completed":
        setStatusMessage(
          `Hoàn thành xử lý file: ${data.filename} (${data.total_logs} logs)`
        );
        setQueuedFiles((prev) => {
          const updated = prev.filter((file) => file !== data.filename);
          console.log(
            "[Simulation] Updated queuedFiles (file_completed):",
            updated
          );
          return updated;
        });
        break;
      case "all_completed":
        setStatusMessage(
          `Hoàn thành tất cả! Tổng cộng đã xử lý ${data.total_logs_processed} logs`
        );
        setIsProcessing(false);
        setCurrentFile("");
        setQueuedFiles([]); // Clear all queued files
        console.log("[Simulation] Cleared queuedFiles (all_completed)");
        break;
      default:
        break;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
            Mô phỏng phân loại giai đoạn tấn công APT thời gian thực
          </h1>
          <p className="text-gray-600 mt-2">
            Phân tích liên tục các tệp log để xác định các giai đoạn tấn công
            APT
          </p>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {statusMessage && (
          <div className="fixed top-4 left-4 max-w-sm p-4 bg-blue-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <FileText className="w-5 h-5" />
            {statusMessage}
          </div>
        )}

        {/* Queue Status Card */}
        {(queueStatus.queue_length > 0 ||
          isProcessing ||
          queuedFiles.length > 0) && (
          <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-lg border border-gray-100 p-6 mb-8 transform transition-all hover:shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Clock className="w-6 h-6 text-blue-600 flex-shrink-0" />
                <div>
                  <h3 className="text-lg font-bold text-gray-900 tracking-tight">
                    Trạng thái xử lý
                  </h3>
                  <p className="text-sm font-medium text-gray-500 mt-1">
                    {isProcessing ? `Đang xử lý: ${currentFile}` : "Đang chờ"}
                  </p>
                </div>
              </div>
            </div>
            {queuedFiles.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm font-semibold text-gray-700 mb-3">
                  Files đang chờ:
                </p>
                <div className="flex flex-wrap gap-2">
                  {queuedFiles.map((filename, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium hover:bg-blue-100 transition-colors duration-200 truncate max-w-xs"
                    >
                      {filename}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
          <div className="p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-4 mb-6">
              <div className="flex-1 min-w-0">
                <input
                  type="file"
                  accept=".csv,.pcap"
                  multiple
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 file:cursor-pointer cursor-pointer"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Có thể chọn nhiều file cùng lúc. Các file sẽ được xử lý tuần
                  tự và liên tục.
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={resetSimulation}
                  className="px-4 py-2 rounded-md text-gray-600 border border-gray-300 hover:bg-gray-50 font-semibold transition-colors duration-200 whitespace-nowrap"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Stats Row */}
            {totalProcessed > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300">
                  <div className="text-3xl font-extrabold text-blue-700 tracking-tight">
                    {totalProcessed}
                  </div>
                  <div className="text-sm font-medium text-blue-900 mt-1">
                    Tổng logs đã xử lý
                  </div>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300">
                  <div className="text-3xl font-extrabold text-purple-700 tracking-tight">
                    {isProcessing ? (
                      <Loader2 className="w-7 h-7 animate-spin text-purple-600" />
                    ) : (
                      <span className="text-green-600">✓</span>
                    )}
                  </div>
                  <div className="text-sm font-medium text-purple-900 mt-1">
                    {isProcessing ? "Đang xử lý" : "Hoàn thành"}
                  </div>
                </div>
              </div>
            )}

            <div className="mb-6">
              <div className="relative h-96">
                <canvas ref={chartRef}></canvas>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold text-gray-800 mb-3">
                Nhật ký dự đoán
              </h2>
              {predictions.length === 0 && !isProcessing ? (
                <div className="flex flex-col items-center gap-4 py-8">
                  <Upload className="w-12 h-12 text-gray-400" />
                  <p className="text-gray-600 text-lg font-medium">
                    Chưa có dự đoán nào
                  </p>
                  <p className="text-gray-500 text-center">
                    Hãy upload các tệp log để bắt đầu phân tích liên tục.
                    <br />
                  </p>
                </div>
              ) : (
                <ul className="max-h-96 overflow-y-auto space-y-2 p-2 rounded-lg bg-gray-50 border border-gray-100">
                  {predictions.map((pred, idx) => (
                    <li
                      key={pred.log_index}
                      className="flex items-center space-x-3 p-3 bg-white rounded-lg border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors duration-200"
                    >
                      <span
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{
                          backgroundColor:
                            phaseColors[pred.stage_label] || phaseColors.Benign,
                        }}
                      ></span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-gray-800 truncate">
                          Log {pred.log_index + 1}: {pred.stage_label}
                        </p>
                        <p className="text-xs text-gray-500 truncate">
                          File: {pred.filename} (Log trong file:{" "}
                          {pred.file_log_index + 1})
                        </p>
                      </div>
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
