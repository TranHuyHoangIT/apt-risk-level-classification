import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { register } from '../services/api';

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    const result = await register(username, email, password);
    if (result.error) {
      if (result.status === 400) {
        if (result.error.includes('Username')) {
          setError('Tên đăng nhập đã tồn tại');
        } else if (result.error.includes('Email')) {
          setError('Email đã tồn tại');
        } else {
          setError('Vui lòng nhập đầy đủ thông tin');
        }
      } else {
        setError(result.error);
      }
      return;
    }

    setSuccess('Đăng ký thành công! Chuyển hướng đến đăng nhập...');
    setTimeout(() => navigate('/login', { replace: true }), 2000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 via-blue-300 to-blue-500">
      <div className="w-full max-w-md bg-white shadow-lg rounded-xl p-8 transform transition duration-500 hover:scale-105">
        <h2 className="text-3xl font-bold text-center text-blue-600 mb-6">
          Đăng ký tài khoản
        </h2>

        {error && (
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded-md text-sm text-center">
            {error}
          </div>
        )}
        {success && (
          <div className="bg-green-100 text-green-700 px-4 py-2 rounded-md text-sm text-center">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="username" className="block font-medium text-gray-700">
              Tên đăng nhập
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none"
              placeholder="Nhập tên đăng nhập"
              required
            />
          </div>

          <div>
            <label htmlFor="email" className="block font-medium text-gray-700">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none"
              placeholder="Nhập email"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block font-medium text-gray-700">
              Mật khẩu
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:outline-none"
              placeholder="Nhập mật khẩu"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full py-3 bg-blue-500 text-white font-bold rounded-lg hover:bg-blue-600 transition duration-300"
          >
            Đăng ký
          </button>
        </form>

        <p className="text-sm text-center mt-5 text-gray-600">
          Đã có tài khoản?{' '}
          <a href="/login" className="text-blue-500 hover:underline font-bold">
            Đăng nhập ngay
          </a>
        </p>
      </div>
    </div>
  );
};

export default Register;
