import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Trash2, AlertCircle, CheckCircle } from 'lucide-react';
import { getUsers, deleteUser } from '../services/api';

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [deleteModal, setDeleteModal] = useState({ open: false, userId: null, username: '' });
  const navigate = useNavigate();

  // Tự động ẩn thông báo sau 3 giây
  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => {
        setError('');
        setSuccess('');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [error, success]);

  // Lấy danh sách người dùng
  useEffect(() => {
    const fetchUsers = async () => {
      const result = await getUsers();
      if (result.error) {
        console.error('[UserManagement] Fetch users error:', result.error);
        setError(result.error);
        if (result.status === 401 || result.status === 403) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/login', { replace: true });
        }
        return;
      }
      console.log('[UserManagement] Fetched users:', result);
      setUsers(result);
    };

    fetchUsers();
  }, [navigate]);

  // Xử lý xóa người dùng
  const handleDelete = async () => {
    const result = await deleteUser(deleteModal.userId);
    if (result.error) {
      console.error('[UserManagement] Delete user error:', result.error);
      setError(result.error);
      setDeleteModal({ open: false, userId: null, username: '' });
      return;
    }
    console.log('[UserManagement] Deleted user:', result);
    setUsers(users.filter((user) => user.id !== deleteModal.userId));
    setSuccess('Xóa người dùng thành công');
    setDeleteModal({ open: false, userId: null, username: '' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
            Quản lý người dùng
          </h1>
          <p className="text-gray-600 mt-2">Xem, chỉnh sửa và xóa thông tin người dùng hệ thống</p>
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

        {/* Modal xác nhận xóa */}
        {deleteModal.open && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 animate-slide-in">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Xác nhận xóa</h3>
              <p className="text-gray-600 mb-6">
                Bạn có chắc muốn xóa người dùng <span className="font-medium">{deleteModal.username}</span>? Hành động này không thể hoàn tác.
              </p>
              <div className="flex justify-end gap-4">
                <button
                  onClick={() => setDeleteModal({ open: false, userId: null, username: '' })}
                  className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-all"
                >
                  Hủy
                </button>
                <button
                  onClick={handleDelete}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Xóa
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Bảng người dùng */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto">
              <thead className="bg-gradient-to-r from-blue-600 to-blue-800 text-white">
                <tr>
                  <th className="py-3 px-4 text-left font-semibold">ID</th>
                  <th className="py-3 px-4 text-left font-semibold">Username</th>
                  <th className="py-3 px-4 text-left font-semibold">Email</th>
                  <th className="py-3 px-4 text-left font-semibold">Role</th>
                  <th className="py-3 px-4 text-left font-semibold">Created At</th>
                  <th className="py-3 px-4 text-left font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="py-4 px-4 text-center text-gray-500">
                      Không có người dùng nào
                    </td>
                  </tr>
                ) : (
                  users.map((user) => (
                    <tr
                      key={user.id}
                      onClick={() => navigate(`/users/${user.id}`)}
                      className="hover:bg-blue-50 transition-all cursor-pointer"
                    >
                      <td className="py-3 px-4 font-medium text-gray-700">{user.id}</td>
                      <td className="py-3 px-4 font-medium text-gray-700">{user.username}</td>
                      <td className="py-3 px-4 font-medium text-gray-700">{user.email}</td>
                      <td className="py-3 px-4 font-medium text-gray-700 capitalize">{user.role}</td>
                      <td className="py-3 px-4 font-medium text-gray-700">
                        {new Date(user.created_at).toLocaleDateString('vi-VN')}
                      </td>
                      <td className="py-3 px-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteModal({ open: true, userId: user.id, username: user.username });
                          }}
                          className="p-2 text-red-600 hover:bg-red-100 rounded-full transition-all"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
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
  );
}