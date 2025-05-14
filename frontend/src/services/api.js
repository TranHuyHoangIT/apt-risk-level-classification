import axios from 'axios';

const BASE_URL = 'http://localhost:5000';

export const uploadLogs = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file); 

    const res = await axios.post(`${BASE_URL}/upload-logs`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return res.data; 
  } catch (error) {
    return { error: error.response ? error.response.data : 'Đã có lỗi xảy ra' };
  }
};

export const uploadPcap = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${BASE_URL}/upload-pcap`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};


export const getUploadHistory = async () => {
  try {
    const res = await axios.get(`${BASE_URL}/upload-history`);
    return res.data;
  } catch (error) {
    return { error: error.response ? error.response.data : 'Đã có lỗi xảy ra' };
  }
};

export const getUploadDetails = async (uploadId) => {
  try {
    const res = await axios.get(`${BASE_URL}/upload-details/${uploadId}`);
    return res.data;
  } catch (error) {
    return { error: error.response ? error.response.data : 'Đã có lỗi xảy ra' };
  }
};

export const getRiskStats = async () => {
  try {
    const res = await axios.get(`${BASE_URL}/risk-stats`);
    return res.data;
  } catch (error) {
    throw new Error(error.response ? error.response.data : 'Failed to fetch risk stats');
  }
};

export const getUploads = async () => {
  try {
    const res = await axios.get(`${BASE_URL}/uploads`);
    return res.data;
  } catch (error) {
    throw new Error(error.response ? error.response.data : 'Failed to fetch uploads');
  }
};


