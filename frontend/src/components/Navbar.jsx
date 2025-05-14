import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="bg-white shadow p-4 flex justify-between items-center">
      <h1 className="text-xl font-bold text-blue-600">APT Risk Dashboard</h1>
      <div>
        <Link to="/" className="mx-2 text-gray-600 hover:text-blue-500">Dashboard</Link>
        <Link to="/upload" className="mx-2 text-gray-600 hover:text-blue-500">Upload Logs</Link>
        <Link to="/upload-history" className="mx-2 text-gray-600 hover:text-blue-500">Risk Details</Link>
      </div>
    </nav>
  );
}
