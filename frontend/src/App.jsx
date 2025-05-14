// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import Dashboard from './pages/Dashboard';
// import UploadLogs from './pages/UploadLogs';
// import RiskDetails from './pages/RiskDetails';
// import NotFound from './pages/NotFound';
// import Navbar from './components/Navbar';
// import Sidebar from './components/Sidebar';

// export default function App() {
//   return (
//     <Router>
//       <div className="flex">
//         <Sidebar />
//         <div className="flex-1 flex flex-col min-h-screen">
//           <Navbar />
//           <div className="p-4">
//             <Routes>
//               <Route path="/" element={<Dashboard />} />
//               <Route path="/upload" element={<UploadLogs />} />
//               <Route path="/details" element={<RiskDetails />} />
//               <Route path="*" element={<NotFound />} />
//             </Routes>
//           </div>
//         </div>
//       </div>
//     </Router>
//   );
// }

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import UploadLogs from './pages/UploadLogs';
import RiskDetails from './pages/RiskDetails';
import UploadHistoryPage from './pages/UploadHistoryPage';  // Import UploadHistoryPage
import NotFound from './pages/NotFound';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

export default function App() {
  return (
    <Router>
      <div className="flex">
        <Sidebar />
        <div className="flex-1 flex flex-col min-h-screen">
          <Navbar />
          <div className="p-4">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/upload" element={<UploadLogs />} />
              <Route path="/upload-history" element={<UploadHistoryPage />} /> {/* Thêm route cho UploadHistoryPage */}
              <Route path="/risk-details/:uploadId" element={<RiskDetails />} /> {/* Cập nhật đường dẫn chi tiết với tham số uploadId */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}

