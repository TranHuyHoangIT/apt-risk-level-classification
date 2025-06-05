import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect, useState } from 'react';

import Dashboard from './pages/Dashboard';
import UploadLogs from './pages/UploadLogs';
import UploadDetails from './pages/UploadDetails';
import UploadHistory from './pages/UploadHistory';
import NotFound from './pages/NotFound';
import Login from './pages/Login';
import Register from './pages/Register';
import Profile from './pages/Profile';
import UserManagement from './pages/UserManagement';
import UserEdit from './pages/UserEdit';
import RealtimeMonitoring from './pages/RealtimeMonitoring'

import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

const ProtectedRoute = ({ children, adminOnly = false }) => {
  const isAuthenticated = !!localStorage.getItem('token');
  const user = localStorage.getItem('user');
  const isAdmin = user && JSON.parse(user).role === 'admin';

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  if (adminOnly && !isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }
  return children;
};

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const checkAuth = () => setIsAuthenticated(!!localStorage.getItem('token'));
    window.addEventListener('storage', checkAuth);
    return () => window.removeEventListener('storage', checkAuth);
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/login" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login onLoginSuccess={() => setIsAuthenticated(true)} />} />
        <Route path="/register" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Register />} />

        <Route path="*" element={
          <div className="flex h-screen overflow-hidden">
            {isAuthenticated && <Sidebar isOpen={sidebarOpen} toggleSidebar={() => setSidebarOpen(!sidebarOpen)} />}
            <div className="flex-1 flex flex-col overflow-y-auto">
              {isAuthenticated && <Navbar toggleSidebar={() => setSidebarOpen(!sidebarOpen)} />}
              <main className="p-4 mt-16 md:mt-14 md:pl-64">
                <Routes>
                  <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                  <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
                  <Route path="/upload" element={<ProtectedRoute><UploadLogs /></ProtectedRoute>} />
                  <Route path="/upload-history" element={<ProtectedRoute><UploadHistory /></ProtectedRoute>} />
                  <Route path="realtime" element={<ProtectedRoute><RealtimeMonitoring /></ProtectedRoute>} />
                  <Route path="/risk-details/:uploadId" element={<ProtectedRoute><UploadDetails /></ProtectedRoute>} />
                  <Route path="/users" element={<ProtectedRoute adminOnly={true}><UserManagement /></ProtectedRoute>} />
                  <Route path="/users/:userId" element={<ProtectedRoute adminOnly={true}><UserEdit /></ProtectedRoute>} />
                  <Route path="/" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </main>
            </div>
          </div>
        }/>
      </Routes>
    </Router>
  );
}