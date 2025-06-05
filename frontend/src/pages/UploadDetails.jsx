import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, AlertCircle, Loader2 } from 'lucide-react';
import { getUploadDetails } from '../services/api';
import LogVisualizer from '../components/LogVisualizer';
import { ALL_TABLE_HEADERS } from '../utils/constants';

export default function UploadDetails() {
  const { uploadId } = useParams();
  const navigate = useNavigate();
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Tự động ẩn thông báo lỗi
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Lấy chi tiết upload
  useEffect(() => {
    const fetchDetails = async () => {
      if (!uploadId) return;
      setLoading(true);
      const data = await getUploadDetails(uploadId);

      if (data.error) {
        console.error('[UploadDetails] Fetch details error:', data.error);
        setError(data.error || 'Đã có lỗi khi lấy chi tiết bản ghi');
        if (data.status === 401 || data.status === 403) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/login', { replace: true });
        }
        setLogs([]);
        setLoading(false);
        return;
      }

      if (Array.isArray(data)) {
        const parsedData = data.map((item) => {
          const values = item.log_data.split(',');
          let row = {};
          ALL_TABLE_HEADERS.forEach((header, index) => {
            row[header] = values[index] || '';
          });
          row.risk_level = item.predicted_label;
          return row;
        });
        setLogs(parsedData);
      } else {
        setLogs([]);
      }
      setLoading(false);
    };

    fetchDetails();
  }, [uploadId, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent flex items-center justify-center gap-3">
            <FileText className="w-8 h-8" />
            Chi tiết bản ghi
          </h1>
          <p className="text-gray-600 mt-2">Xem chi tiết các log và mức độ rủi ro của file upload</p>
        </div>

        {/* Thông báo lỗi */}
        {error && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* Card chính */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
          <div className="p-4 sm:p-6">
            {!uploadId ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <FileText className="w-16 h-16 text-gray-400" />
                <p className="text-gray-600 text-lg font-medium">Vui lòng chọn một upload để xem chi tiết</p>
                <p className="text-gray-500">Quay lại trang lịch sử upload để chọn file.</p>
              </div>
            ) : loading ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                <p className="text-gray-600 font-medium">Đang tải chi tiết bản ghi...</p>
              </div>
            ) : logs.length === 0 ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <FileText className="w-16 h-16 text-gray-400" />
                <p className="text-gray-600 text-lg font-medium">Không có dữ liệu để hiển thị</p>
                <p className="text-gray-500">File upload có thể trống hoặc không hợp lệ.</p>
              </div>
            ) : (
              <LogVisualizer logs={logs} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}