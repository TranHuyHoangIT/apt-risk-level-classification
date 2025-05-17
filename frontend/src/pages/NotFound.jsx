import { useNavigate } from 'react-router-dom';
import { AlertTriangle, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200 flex items-center justify-center px-4 sm:px-6">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl border border-gray-100 p-6 sm:p-8 transform transition-all hover:scale-[1.01] animate-slide-in">
        {/* Icon lỗi */}
        <div className="flex justify-center mb-6">
          <div className="relative flex items-center justify-center w-24 h-24 sm:w-28 sm:h-28 bg-white rounded-full border-4 border-blue-600">
            <AlertTriangle className="w-16 h-16 sm:w-20 sm:h-20 text-blue-600 animate-pulse" />
          </div>
        </div>

        {/* Tiêu đề */}
        <h1 className="text-6xl sm:text-8xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent text-center mb-4">
          404
        </h1>

        {/* Mô tả */}
        <p className="text-lg sm:text-xl text-gray-600 text-center mb-2">
          Trang bạn tìm không tồn tại
        </p>
        <p className="text-gray-500 text-center mb-6">
          Hãy kiểm tra URL hoặc quay lại trang chủ
        </p>

        {/* Nút quay lại */}
        <div className="flex justify-center">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white py-2 px-4 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all transform hover:scale-105"
          >
            <ArrowLeft className="w-5 h-5" />
            Quay lại trang chủ
          </button>
        </div>
      </div>
    </div>
  );
}