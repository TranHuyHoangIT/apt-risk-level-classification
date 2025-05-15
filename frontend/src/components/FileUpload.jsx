import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';

export default function FileUpload({ onUpload, accept = '.csv,.pcap' }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validExtensions = ['.csv', '.pcap'];
      const fileExtension = selectedFile.name.slice(selectedFile.name.lastIndexOf('.')).toLowerCase();
      if (!validExtensions.includes(fileExtension)) {
        setError('Vui lòng chọn file .csv hoặc .pcap!');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError(null);
    }
  };

  const handleSubmit = () => {
    if (file) {
      onUpload(file);
      setFile(null);
      setError(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } else {
      setError('Vui lòng chọn một file để upload!');
    }
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-lg max-w-md mx-auto text-center">
      <label
        htmlFor="file-upload"
        className={`cursor-pointer flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed rounded-2xl transition ${
          error ? 'border-red-500 hover:border-red-600' : 'border-gray-300 hover:border-blue-500'
        }`}
      >
        <UploadCloud className="w-6 h-6 text-gray-500" />
        <span className="text-gray-600">Chọn file CSV hoặc PCAP để upload</span>
      </label>
      <input
        id="file-upload"
        type="file"
        accept={accept}
        onChange={handleChange}
        ref={fileInputRef}
        className="hidden"
      />

      {file && (
        <p className="mt-3 text-gray-700 text-sm">
          📄 <strong>{file.name}</strong>
        </p>
      )}

      <button
        onClick={handleSubmit}
        disabled={!file}
        className="mt-5 w-full px-4 py-3 bg-blue-600 text-white rounded-2xl hover:bg-blue-700 transition font-semibold disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        Tải lên
      </button>

      {error && (
        <p className="mt-3 text-sm text-red-600">{error}</p>
      )}

      <p className="mt-2 text-sm text-gray-500">Chấp nhận các tệp .csv hoặc .pcap</p>
    </div>
  );
}
