import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Save, AlertCircle, CheckCircle } from 'lucide-react';
import { getUser, updateUser } from '../services/api';

export default function UserEdit() {
  const { userId } = useParams();
  const [user, setUser] = useState(null);
  const [form, setForm] = useState({ username: '', email: '', newPassword: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  // Tự động ẩn thông báo
  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => {
        setError('');
        setSuccess('');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [error, success]);

  // Lấy thông tin người dùng
  useEffect(() => {
    const fetchUser = async () => {
      const result = await getUser(userId);
      if (result.error) {
        console.error('[UserEdit] Fetch user error:', result.error);
        setError(result.error);
        if (result.status === 401 || result.status === 403) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/login', { replace: true });
        }
        return;
      }
      console.log('[UserEdit] Fetched user:', result);
      setUser(result);
      setForm({ username: result.username, email: result.email, newPassword: '' });
    };

    fetchUser();
  }, [userId, navigate]);

  // Xử lý thay đổi input
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // Xử lý submit form
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    const updateData = {
      username: form.username,
      email: form.email,
    };
    if (form.newPassword) {
      updateData.newPassword = form.newPassword;
    }

    const result = await updateUser(userId, updateData);
    if (result.error) {
      console.error('[UserEdit] Update user error:', result.error);
      setError(result.error);
      return;
    }
    console.log('[UserEdit] Updated user:', result);
    setUser(result);
    setSuccess('Cập nhật người dùng thành công');
    setForm({ ...form, newPassword: '' });
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-600 font-medium">Đang tải thông tin người dùng...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
            Chỉnh sửa người dùng
          </h1>
          <p className="text-gray-600 mt-2">Quản lý thông tin của người dùng</p>
        </div>

        {/* Thông báo */}
        {error && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-red-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}
        {success && (
          <div className="fixed top-4 right-4 max-w-sm p-4 bg-green-600 text-white rounded-lg shadow-lg flex items-center gap-2 animate-slide-in z-50">
            <CheckCircle className="w-5 h-5" />
            {success}
          </div>
        )}

        {/* Card chính */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden transform transition-all hover:scale-[1.01] duration-300">
          {/* Avatar và thông tin tổng quan */}
          <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-8 flex flex-col items-center">
            <div className="w-24 h-24 bg-white text-blue-600 flex items-center justify-center rounded-full text-4xl font-bold mb-4 shadow-md">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <h2 className="text-2xl font-semibold text-white">{user.username}</h2>
            <p className="text-blue-200">{user.email}</p>
            <p className="text-blue-200">Vai trò: {user.role}</p>
          </div>

          <div className="p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Username */}
              <div className="relative">
                <input
                  type="text"
                  name="username"
                  value={form.username}
                  onChange={handleChange}
                  className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                  required
                  placeholder=" "
                />
                <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                  Tên người dùng
                </label>
              </div>

              {/* Email */}
              <div className="relative">
                <input
                  type="email"
                  name="email"
                  value={form.email}
                  onChange={handleChange}
                  className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                  required
                  placeholder=" "
                />
                <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                  Email
                </label>
              </div>

              {/* New Password */}
              <div className="relative">
                <input
                  type="password"
                  name="newPassword"
                  value={form.newPassword}
                  onChange={handleChange}
                  className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                  placeholder=" "
                />
                <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                  Mật khẩu mới (để trống nếu không đổi)
                </label>
              </div>

              {/* Submit */}
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all flex items-center justify-center gap-2 group"
              >
                <Save className="w-4 h-4 transform group-hover:scale-110 transition-transform" />
                Lưu thay đổi
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}