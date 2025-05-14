import React, { useState } from 'react';
import { UploadCloud } from 'lucide-react'; // Thêm icon hiện đại nếu có lucide-react

export default function FileUpload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = () => {
    if (file) {
      onUpload(file);
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
        <span className="text-gray-600">Chọn file để upload</span>
      </label>
      <input
        id="file-upload"
        type="file"
        onChange={handleChange}
        className="hidden"
      />

      {file && (
        <p className="mt-3 text-gray-700 text-sm">
          📄 <strong>{file.name}</strong>
        </p>
      )}

      <button
        onClick={handleSubmit}
        className="mt-5 w-full px-4 py-3 bg-blue-600 text-white rounded-2xl hover:bg-blue-700 transition font-semibold"
      >
        Tải lên
      </button>

      {error && (
        <p className="mt-3 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}
