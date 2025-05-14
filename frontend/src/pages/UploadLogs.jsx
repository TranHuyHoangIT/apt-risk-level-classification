import { useState } from 'react';
import Papa from 'papaparse';
import FileUpload from '../components/FileUpload';
import { uploadLogs } from '../services/api';
import Loader from '../components/Loader';
import LogVisualizer from '../components/LogVisualizer';
import { ALL_TABLE_HEADERS } from '../utils/constants'; // Đường dẫn đến file constants.js

export default function UploadLogs() {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [results, setResults] = useState([]);

  const handleUpload = async (file) => {
    setUploading(true);

    Papa.parse(file, {
      header: false,
      skipEmptyLines: true,
      complete: async (parsedData) => {
        const originalData = parsedData.data;

        const result = await uploadLogs(file);

        if (result.results) {
          const mergedResults = result.results.map((res) => {
            const logIndex = res.log_index;
            const logRow = originalData[logIndex] || [];

            let row = {};
            ALL_TABLE_HEADERS.forEach((header, index) => {
              row[header] = logRow[index] || '';
            });
            row.risk_level = res.risk_level;

            return row;
          });

          setResults(mergedResults);
          setMessage('Upload và phân loại thành công!');
        } else {
          setMessage('Lỗi khi phân loại dữ liệu.');
        }

        setUploading(false);
      },
      error: () => {
        setMessage('Lỗi khi đọc file CSV.');
        setUploading(false);
      },
    });
  };

  return (
    <div className="p-6 bg-gray-50 rounded-lg shadow-md">
      <div className="text-center mb-6">
        <h2 className="text-3xl font-bold text-blue-600">Upload Logs</h2>
        <p className="text-gray-500">Tải lên tệp log để phân tích và hiển thị kết quả</p>
      </div>

      <FileUpload onUpload={handleUpload} />
      {uploading && <Loader />}
      {message && <p className="mt-4 text-green-600 text-center">{message}</p>}

      {results.length > 0 && <LogVisualizer logs={results} />}
    </div>
  );
}