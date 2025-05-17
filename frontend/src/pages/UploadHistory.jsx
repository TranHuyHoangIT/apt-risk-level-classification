import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, AlertCircle, Loader2 } from 'lucide-react';
import { getUploadHistory } from '../services/api';

export default function UploadHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Tự động ẩn thông báo lỗi
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Lấy lịch sử upload
  useEffect(() => {
    async function fetchHistory() {
      setLoading(true);
      const res = await getUploadHistory();
      if (res.error) {
        console.error('[UploadHistory] Fetch history error:', res.error);
        setError(res.error || 'Đã có lỗi khi lấy lịch sử upload');
        if (res.status === 401 || res.status === 403) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/login', { replace: true });
        }
        setHistory([]);
        setLoading(false);
        return;
      }
      console.log('[UploadHistory] Fetched history:', res);
      setHistory(res || []);
      setLoading(false);
    }
    fetchHistory();
  }, [navigate]);

  // Xử lý chọn upload
  const handleSelectUpload = (uploadId) => {
    navigate(`/risk-details/${uploadId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent flex items-center justify-center gap-3">
            <History className="w-8 h-8" />
            Lịch sử Upload
          </h1>
          <p className="text-gray-600 mt-2">Xem lại các file log đã upload và chi tiết rủi ro</p>
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
          {/* Card content */}
          <div className="p-4 sm:p-6">
            {loading ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                <p className="text-gray-600 font-medium">Đang tải lịch sử upload...</p>
              </div>
            ) : history.length === 0 ? (
              <div className="flex flex-col items-center gap-4 py-8">
                <History className="w-16 h-16 text-gray-400" />
                <p className="text-gray-600 text-lg font-medium">Chưa có upload nào</p>
                <p className="text-gray-500">Hãy upload file log để bắt đầu phân tích rủi ro.</p>
              </div>
            ) : (
              <>
                <p className="text-gray-600 mb-4">Tổng số lần upload: {history.length}</p>
                <div className="overflow-x-auto">
                  <table className="min-w-full table-auto">
                    <thead className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
                      <tr>
                        <th className="py-3 px-4 text-left font-semibold">Upload ID</th>
                        <th className="py-3 px-4 text-left font-semibold">Filename</th>
                        <th className="py-3 px-4 text-left font-semibold">Upload Time</th>
                        <th className="py-3 px-4 text-right font-semibold">Tổng số log</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((item, index) => (
                        <tr
                          key={item.upload_id}
                          onClick={() => handleSelectUpload(item.upload_id)}
                          className={`hover:bg-blue-50 transition-all cursor-pointer ${
                            index % 2 === 0 ? 'bg-gray-50' : 'bg-white'
                          }`}
                        >
                          <td className="py-3 px-4 font-medium text-gray-700">{item.upload_id}</td>
                          <td className="py-3 px-4 font-medium text-gray-700">{item.filename}</td>
                          <td className="py-3 px-4 font-medium text-gray-700">
                            {new Date(item.upload_time).toLocaleString('vi-VN')}
                          </td>
                          <td className="py-3 px-4 font-medium text-gray-700 text-right">
                            {item.total_logs}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}