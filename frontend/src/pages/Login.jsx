import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../services/api';

const Login = ({ onLoginSuccess }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

     const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        const result = await login(username, password);

        if (result.error) {
            if (result.status === 401) {
                setError('Tên đăng nhập hoặc mật khẩu không đúng');
            } else if (result.status === 400) {
                setError('Vui lòng nhập đầy đủ thông tin');
            } else {
                setError(result.error);
            }
            return;
        }

        localStorage.setItem('token', result.access_token);
        localStorage.setItem('user', JSON.stringify(result.user));
        
        if (onLoginSuccess) {
            onLoginSuccess();
        }

        navigate('/dashboard', { replace: true });
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500">
            <div className="w-full max-w-md bg-white shadow-2xl rounded-lg p-8 transform transition duration-500 hover:scale-105">
                <h2 className="text-3xl font-bold text-center text-blue-600 mb-6">
                    Đăng nhập hệ thống
                </h2>

                {error && (
                    <div className="bg-red-200 text-red-700 px-4 py-2 rounded-md text-sm text-center">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="username" className="block font-semibold text-gray-700">
                            Tên đăng nhập
                        </label>
                        <input
                            id="username"
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            placeholder="Nhập tên đăng nhập"
                            required
                        />
                    </div>

                    <div>
                        <label htmlFor="password" className="block font-semibold text-gray-700">
                            Mật khẩu
                        </label>
                        <input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            placeholder="Nhập mật khẩu"
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        className="w-full py-3 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700 transition duration-300"
                    >
                        Đăng nhập
                    </button>
                </form>

                <p className="text-sm text-center mt-5 text-gray-600">
                    Chưa có tài khoản?{' '}
                    <a href="/register" className="text-indigo-500 hover:underline font-bold">
                        Đăng ký ngay
                    </a>
                </p>
            </div>
        </div>
    );
};

export default Login;
