import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Mail, Key, Save, AlertCircle, CheckCircle } from 'lucide-react';
import { getProfile, updateProfile, changePassword } from '../services/api';

export default function Profile() {
    const [user, setUser] = useState(null);
    const [profileForm, setProfileForm] = useState({ username: '', email: '' });
    const [passwordForm, setPasswordForm] = useState({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
    });
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
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

    // Kiểm tra đăng nhập và lấy thông tin user
    useEffect(() => {
        const token = localStorage.getItem('token');
        if (!token) {
            navigate('/login', { replace: true });
            return;
        }

        const fetchProfile = async () => {
            const result = await getProfile();
            if (result.error) {
                console.error('[Profile] Fetch profile error:', result.error);
                setError(result.error);
                if (result.status === 401) {
                    localStorage.removeItem('token');
                    localStorage.removeItem('user');
                    navigate('/login', { replace: true });
                }
                return;
            }
            console.log('[Profile] Fetched profile:', result);
            setUser(result);
            setProfileForm({ username: result.username, email: result.email });
        };

        fetchProfile();
    }, [navigate]);

    // Xử lý cập nhật thông tin
    const handleProfileSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        const result = await updateProfile(profileForm);
        if (result.error) {
            console.error('[Profile] Update profile error:', result.error);
            setError(result.error);
            return;
        }
        console.log('[Profile] Updated profile:', result);
        setUser(result);
        localStorage.setItem('user', JSON.stringify(result));
        window.dispatchEvent(new Event('storage'));
        setSuccess('Cập nhật thông tin thành công');
    };

    // Xử lý đổi mật khẩu
    const handlePasswordSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (passwordForm.newPassword !== passwordForm.confirmPassword) {
            setError('Mật khẩu mới và xác nhận không khớp');
            return;
        }

        const result = await changePassword({
            currentPassword: passwordForm.currentPassword,
            newPassword: passwordForm.newPassword,
        });
        if (result.error) {
            console.error('[Profile] Change password error:', result.error);
            setError(result.error);
            return;
        }
        console.log('[Profile] Changed password:', result);
        setSuccess('Đổi mật khẩu thành công');
        setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
    };

    // Xử lý thay đổi input
    const handleProfileChange = (e) => {
        setProfileForm({ ...profileForm, [e.target.name]: e.target.value });
    };

    const handlePasswordChange = (e) => {
        setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });
    };

    // Hàm lấy chữ cái đầu cho avatar
    const getInitial = (username) => {
        return username ? username.charAt(0).toUpperCase() : '?';
    };

    if (!user) {
        return (
            <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-200 flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-gray-600 font-medium">Đang tải hồ sơ...</p>
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
                        Hồ sơ người dùng
                    </h1>
                    <p className="text-gray-600 mt-2">Quản lý thông tin cá nhân và bảo mật tài khoản</p>
                </div>

                {/* Thông báo dạng toast */}
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
                            {getInitial(user.username)}
                        </div>
                        <h2 className="text-2xl font-semibold text-white">{user.username}</h2>
                        <p className="text-blue-200">{user.email}</p>
                    </div>

                    <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Thông tin user */}
                        <div className="space-y-6">
                            <h3 className="text-xl font-semibold text-gray-800">Thông tin cá nhân</h3>
                            <div className="space-y-4">
                                <div className="flex items-center gap-3">
                                    <User className="w-5 h-5 text-blue-600" />
                                    <div>
                                        <p className="text-sm text-gray-500">Tên người dùng</p>
                                        <p className="text-gray-800 font-medium">{user.username}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Mail className="w-5 h-5 text-blue-600" />
                                    <div>
                                        <p className="text-sm text-gray-500">Email</p>
                                        <p className="text-gray-800 font-medium">{user.email}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Key className="w-5 h-5 text-blue-600" />
                                    <div>
                                        <p className="text-sm text-gray-500">Vai trò</p>
                                        <p className="text-gray-800 font-medium capitalize">{user.role}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div>
                                        <p className="text-sm text-gray-500">Ngày tạo</p>
                                        <p className="text-gray-800 font-medium">
                                            {new Date(user.created_at).toLocaleDateString('vi-VN')}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Form cập nhật thông tin */}
                        <div className="space-y-6">
                            <h3 className="text-xl font-semibold text-gray-800">Cập nhật thông tin</h3>
                            <form onSubmit={handleProfileSubmit} className="space-y-4">
                                <div className="relative">
                                    <input
                                        type="text"
                                        name="username"
                                        value={profileForm.username}
                                        onChange={handleProfileChange}
                                        className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                                        required
                                        placeholder=" "
                                    />
                                    <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                                        Tên người dùng
                                    </label>
                                </div>
                                <div className="relative">
                                    <input
                                        type="email"
                                        name="email"
                                        value={profileForm.email}
                                        onChange={handleProfileChange}
                                        className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                                        required
                                        placeholder=" "
                                    />
                                    <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                                        Email
                                    </label>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all flex items-center justify-center gap-2 group"
                                >
                                    <Save className="w-4 h-4 transform group-hover:scale-110 transition-transform" />
                                    Lưu thay đổi
                                </button>
                            </form>
                        </div>

                        {/* Form đổi mật khẩu */}
                        <div className="space-y-6 md:col-span-2">
                            <h3 className="text-xl font-semibold text-gray-800">Đổi mật khẩu</h3>
                            <form onSubmit={handlePasswordSubmit} className="space-y-4">
                                <div className="relative">
                                    <input
                                        type="password"
                                        name="currentPassword"
                                        value={passwordForm.currentPassword}
                                        onChange={handlePasswordChange}
                                        className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                                        required
                                        placeholder=" "
                                    />
                                    <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                                        Mật khẩu hiện tại
                                    </label>
                                </div>
                                <div className="relative">
                                    <input
                                        type="password"
                                        name="newPassword"
                                        value={passwordForm.newPassword}
                                        onChange={handlePasswordChange}
                                        className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                                        required
                                        placeholder=" "
                                    />
                                    <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                                        Mật khẩu mới
                                    </label>
                                </div>
                                <div className="relative">
                                    <input
                                        type="password"
                                        name="confirmPassword"
                                        value={passwordForm.confirmPassword}
                                        onChange={handlePasswordChange}
                                        className="w-full p-3 pt-5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all peer"
                                        required
                                        placeholder=" "
                                    />
                                    <label className="absolute top-0 left-3 text-sm text-gray-500 transform -translate-y-1 scale-75 origin-top peer-placeholder-shown:translate-y-4 peer-placeholder-shown:scale-100 peer-focus:-translate-y-1 peer-focus:scale-75 transition-all duration-200">
                                        Xác nhận mật khẩu mới
                                    </label>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all flex items-center justify-center gap-2 group"
                                >
                                    <Key className="w-4 h-4 transform group-hover:scale-110 transition-transform" />
                                    Đổi mật khẩu
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}