import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText } from 'lucide-react';
import { getStageStats, getUploads } from '../services/api';
import StageChart from '../components/StageChart';
import Loader from '../components/Loader';

const STAGE_ORDER = ['Benign', 'Reconnaissance', 'Establish Foothold', 'Lateral Movement', 'Data Exfiltration'];

const stageColors = {
  'Benign': 'bg-gray-500',
  'Reconnaissance': 'bg-lime-500',
  'Establish Foothold': 'bg-yellow-400',
  'Lateral Movement': 'bg-orange-500',
  'Data Exfiltration': 'bg-red-600',
};

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [uploads, setUploads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/login', { replace: true });
      return;
    }

    async function fetchData() {
      try {
        const [statsData, uploadsData] = await Promise.all([
          getStageStats(),
          getUploads(),
        ]);

        const processedStageOverview = STAGE_ORDER.map((stageLabel) => {
          const item = statsData.stage_overview.find((i) => i.stage_label === stageLabel);
          return {
            stage_label: stageLabel,
            count: item ? Number(item.count) : 0,
          };
        });

        setStats({ ...statsData, stage_overview: processedStageOverview });
        setUploads(uploadsData);
        setError('');
      } catch (err) {
        if (err.response?.status === 401) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/login', { replace: true });
          setError('Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.');
        } else if (err.response?.status === 403) {
          setError('Bạn không có quyền truy cập dữ liệu này.');
        } else {
          setError(err.response?.data?.error || 'Lỗi khi tải dữ liệu');
        }
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [navigate]);

  if (loading) return <Loader />;

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Thông báo lỗi dạng toast */}
        {error && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            {error}
          </div>
        )}

        {/* Phân bố giai đoạn tấn công */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent mb-2">
            Phân bố giai đoạn tấn công APT
          </h2>
          <p className="text-gray-600 mb-6">Tổng quan giai đoạn tấn công từ các log đã upload</p>
          {stats && (
            <div className="bg-white p-6 rounded-xl shadow-xl border border-blue-600/10 transform transition-all hover:scale-[1.01] duration-300">
              <StageChart data={stats.stage_overview} className="w-full h-80" />
            </div>
          )}
        </div>

        {/* Lịch sử upload file */}
        <div>
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent mb-2">
            Lịch sử upload file
          </h2>
          <p className="text-gray-600 mb-6">Danh sách các file log đã được upload và phân tích</p>
          <div className="bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
            <div className="overflow-x-auto min-w-0">
              <table className="min-w-full table-auto">
                <thead className="bg-gradient-to-r from-blue-600 to-blue-800 text-white sticky top-0">
                  <tr>
                    <th className="px-6 py-3 text-center w-12">#</th>
                    <th className="px-6 py-3 text-left min-w-[150px]">Tên file</th>
                    <th className="px-6 py-3 text-center min-w-[160px]">Thời gian upload</th>
                    <th className="px-6 py-3 text-center w-32">Tổng số log</th>
                    <th className="px-6 py-3 text-left min-w-[500px]">Tóm tắt giai đoạn tấn công</th>
                  </tr>
                </thead>
                <tbody>
                  {uploads.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-8 text-center text-gray-500">
                        <div className="flex flex-col items-center gap-4">
                          <FileText className="w-16 h-16 text-gray-400" />
                          <p className="text-gray-600 text-lg font-medium">Chưa có file nào được upload</p>
                          <p className="text-gray-500">Hãy upload file log để bắt đầu phân tích</p>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    uploads.map((upload, idx) => (
                      <tr key={upload.upload_id} className="hover:bg-blue-50 transition-all">
                        <td className="px-6 py-4 text-center font-medium">{idx + 1}</td>
                        <td className="px-6 py-4 font-medium">{upload.filename}</td>
                        <td className="px-6 py-4 text-center text-gray-600">
                          {new Date(upload.upload_time).toLocaleString('vi-VN')}
                        </td>
                        <td className="px-6 py-4 text-center font-medium">{upload.total_logs}</td>
                        <td className="px-6 py-4">
                          <div className="flex flex-wrap gap-2 flex-row">
                            {STAGE_ORDER.map((stageLabel, i) => {
                              const item = upload.stage_summary.find(
                                (rs) => rs.stage_label === stageLabel
                              );
                              const count = item ? item.count : 0;
                              return (
                                <span
                                  key={i}
                                  className={`px-2 py-1 rounded text-xs text-white font-medium ${stageColors[stageLabel]}`}
                                >
                                  {stageLabel}: {count}
                                </span>
                              );
                            })}
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}