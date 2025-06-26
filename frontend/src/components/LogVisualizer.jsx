import { Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { ALL_TABLE_HEADERS, STAGE_LABELS, CHART_COLORS} from '../utils/constants';

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  ChartDataLabels
);

export default function LogVisualizer({ logs }) {
  // Thống kê dữ liệu rủi ro cho chart
  const getStageStats = () => {
    const stageCounts = { ...STAGE_LABELS };
    logs.forEach((log) => {
      if (log.stage_label in stageCounts) {
        stageCounts[log.stage_label] += 1;
      }
    });
    return stageCounts;
  };

  const stageCounts = getStageStats();
  const totalLogs = Object.values(stageCounts).reduce((a, b) => a + b, 0);

  const pieData = {
    labels: Object.keys(stageCounts),
    datasets: [
      {
        data: Object.values(stageCounts),
        backgroundColor: CHART_COLORS,
        hoverOffset: 4,
      },
    ],
  };

  const pieOptions = {
    plugins: {
      datalabels: {
        formatter: (value) => {
          if (totalLogs === 0) return '0%';
          const percentage = ((value / totalLogs) * 100).toFixed(1);
          return `${percentage}%`;
        },
        color: '#fff',
        font: {
          weight: 'bold',
          size: 14,
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const count = context.parsed;
            const percentage = ((count / totalLogs) * 100).toFixed(1);
            return `${context.label}: ${count} (${percentage}%)`;
          },
        },
      },
    },
    aspectRatio: 1.5,
  };

  const barData = {
    labels: Object.keys(stageCounts),
    datasets: [
      {
        label: 'Số lượng log',
        data: Object.values(stageCounts),
        backgroundColor: CHART_COLORS,
        borderColor: CHART_COLORS,
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="mt-6">
      <h3 className="text-xl font-semibold text-center text-gray-700">📊 Kết quả phân loại</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-4">
        <div className="p-4 bg-white rounded-xl shadow-md">
          <h4 className="text-lg font-semibold mb-2 text-gray-700">Tỷ lệ giai đoạn tấn công</h4>
          <Pie data={pieData} options={pieOptions} />
        </div>
        <div className="p-4 bg-white rounded-xl shadow-md">
          <h4 className="text-lg font-semibold mb-2 text-gray-700">Số lượng các giai đoạn tấn công</h4>
          <Bar data={barData} options={{ aspectRatio: 1.5 }} />
        </div>
      </div>

      <div className="mt-8">
        <h4 className="text-lg font-semibold text-gray-700">📑 Dữ liệu dự đoán chi tiết</h4>
        <div className="max-w-[1200px] overflow-x-auto max-h-[600px] bg-white rounded-xl shadow-md">
          <table className="w-full text-sm text-left text-gray-700 border">
            <thead className="bg-gray-100 sticky top-0 z-10">
              <tr>
                {ALL_TABLE_HEADERS.map((col) => (
                  <th key={col} className="py-2 px-3 border text-gray-600 whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logs.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-gray-50">
                  {ALL_TABLE_HEADERS.map((col, colIndex) => (
                    <td key={colIndex} className="py-1 px-3 border whitespace-nowrap">
                      {row[col] != null ? row[col].toString() : ''}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}