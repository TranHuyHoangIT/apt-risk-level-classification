import { NavLink } from 'react-router-dom';
import { ShieldCheck, Upload, List, Users } from 'lucide-react';

export default function Sidebar({ isOpen, toggleSidebar }) {
  const user = localStorage.getItem('user');
  const isAdmin = user && JSON.parse(user).role === 'admin';

  return (
    <aside
      className={`fixed left-0 top-0 h-screen w-64 z-50 bg-gradient-to-b from-gray-900 to-gray-800 shadow-xl transition-transform duration-300 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      } md:translate-x-0`}
    >
      {/* Logo */}
      <div className="p-4 sm:p-6 flex items-center gap-2">
        <ShieldCheck className="w-8 h-8 text-blue-400" />
        <span className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
          APT Stage
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col px-4 py-2 space-y-2">
        <NavLink
          to="/dashboard"
          onClick={() => window.innerWidth < 768 && toggleSidebar()}
          className={({ isActive }) =>
            `flex items-center gap-3 p-3 rounded-lg text-gray-200 hover:bg-blue-700 hover:text-white transition-all duration-200 group ${
              isActive ? 'bg-blue-700 text-white' : ''
            }`
          }
        >
          <ShieldCheck className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
          Dashboard
        </NavLink>
        <NavLink
          to="/upload"
          onClick={() => window.innerWidth < 768 && toggleSidebar()}
          className={({ isActive }) =>
            `flex items-center gap-3 p-3 rounded-lg text-gray-200 hover:bg-blue-700 hover:text-white transition-all duration-200 group ${
              isActive ? 'bg-blue-700 text-white' : ''
            }`
          }
        >
          <Upload className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
          Upload Logs
        </NavLink>
        <NavLink
          to="/upload-history"
          onClick={() => window.innerWidth < 768 && toggleSidebar()}
          className={({ isActive }) =>
            `flex items-center gap-3 p-3 rounded-lg text-gray-200 hover:bg-blue-700 hover:text-white transition-all duration-200 group ${
              isActive ? 'bg-blue-700 text-white' : ''
            }`
          }
        >
          <List className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
          Upload Archive
        </NavLink>
        {isAdmin && (
          <NavLink
            to="/users"
            onClick={() => window.innerWidth < 768 && toggleSidebar()}
            className={({ isActive }) =>
              `flex items-center gap-3 p-3 rounded-lg text-gray-200 hover:bg-blue-700 hover:text-white transition-all duration-200 group ${
                isActive ? 'bg-blue-700 text-white' : ''
              }`
            }
          >
            <Users className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
            User Management
          </NavLink>
        )}
      </nav>
    </aside>
  );
}