import { Link } from 'react-router-dom';
import { ShieldCheck, Upload, List } from 'lucide-react';

export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-100 min-h-screen shadow">
      <div className="p-4 font-bold text-blue-600">APT Risk</div>
      <nav className="flex flex-col p-4 space-y-2">
        <Link to="/" className="flex items-center text-gray-700 hover:bg-blue-100 p-2 rounded">
          <ShieldCheck className="w-5 h-5 mr-2" /> Dashboard
        </Link>
        <Link to="/upload" className="flex items-center text-gray-700 hover:bg-blue-100 p-2 rounded">
          <Upload className="w-5 h-5 mr-2" /> Upload Logs
        </Link>
        <Link to="/upload-history" className="flex items-center text-gray-700 hover:bg-blue-100 p-2 rounded">
          <List className="w-5 h-5 mr-2" /> Risk Details
        </Link>
      </nav>
    </aside>
  );
}
