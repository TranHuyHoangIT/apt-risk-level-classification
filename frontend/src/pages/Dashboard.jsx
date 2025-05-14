import { useEffect, useState } from 'react';
import { getRiskStats, getUploads } from '../services/api';
import RiskChart from '../components/RiskChart';
import Loader from '../components/Loader';

const RISK_ORDER = ['Không có', 'Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao'];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [uploads, setUploads] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsData, uploadsData] = await Promise.all([
          getRiskStats(),
          getUploads()
        ]);

        const processedRiskOverview = RISK_ORDER.map(riskLevel => {
          const item = statsData.risk_overview.find(i => i.risk_level === riskLevel);
          return {
            risk_level: riskLevel,
            count: item ? Number(item.count) : 0,
          };
        });

        setStats({ ...statsData, risk_overview: processedRiskOverview });
        setUploads(uploadsData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <Loader />;

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Phân bố mức độ rủi ro</h2>
      <RiskChart data={stats.risk_overview} />

      <h2 className="text-2xl font-bold mt-10 mb-4">Lịch sử upload file</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white shadow rounded-lg table-auto">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 border-b text-center w-12">#</th>
              <th className="px-4 py-2 border-b text-left min-w-[200px]">Tên file</th>
              <th className="px-4 py-2 border-b text-center min-w-[160px]">Thời gian upload</th>
              <th className="px-4 py-2 border-b text-center w-32">Tổng số log</th>
              <th className="px-4 py-2 border-b text-left min-w-[300px]">Tóm tắt rủi ro</th>
            </tr>
          </thead>
          <tbody>
            {uploads.map((upload, idx) => (
              <tr key={upload.upload_id} className="hover:bg-gray-50">
                <td className="px-4 py-2 border-b text-center">{idx + 1}</td>
                <td className="px-4 py-2 border-b">{upload.filename}</td>
                <td className="px-4 py-2 border-b text-center">{upload.upload_time}</td>
                <td className="px-4 py-2 border-b text-center">{upload.total_logs}</td>
                <td className="px-4 py-2 border-b">
                  <div className="flex flex-wrap gap-2">
                    {RISK_ORDER.map((riskLevel, i) => {
                      const item = upload.risk_summary.find(rs => rs.risk_level === riskLevel);
                      const count = item ? item.count : 0;
                      return (
                        <span
                          key={i}
                          className="bg-gray-100 px-2 py-1 rounded text-xs text-gray-700"
                        >
                          {riskLevel}: {count}
                        </span>
                      );
                    })}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
