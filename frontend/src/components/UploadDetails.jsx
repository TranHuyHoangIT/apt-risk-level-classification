import { useEffect, useState } from 'react';
import { getUploadDetails } from '../services/api';
import LogVisualizer from '../components/LogVisualizer'; 
import { ALL_TABLE_HEADERS } from '../utils/constants';

export default function UploadDetails({ uploadId }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchDetails = async () => {
      if (!uploadId) return;
      setLoading(true);
      const data = await getUploadDetails(uploadId);

      if (Array.isArray(data)) {
        const parsedData = data.map((item) => {
          const values = item.log_data.split(',');
          let row = {};
          ALL_TABLE_HEADERS.forEach((header, index) => {
            row[header] = values[index] || '';
          });
          row.risk_level = item.predicted_label;
          return row;
        });
        setLogs(parsedData);
      } else {
        setLogs([]);
      }
      setLoading(false);
    };

    fetchDetails();
  }, [uploadId]);

  if (!uploadId) {
    return <p className="text-center text-gray-500">Vui lòng chọn một upload để xem chi tiết.</p>;
  }

  if (loading) {
    return <p className="text-center text-gray-500">Đang tải dữ liệu...</p>;
  }

  if (logs.length === 0) {
    return <p className="text-center text-gray-500">Không có dữ liệu để hiển thị.</p>;
  }

  return (
    <div className="p-4">
      <LogVisualizer logs={logs} />
    </div>
  );
}
