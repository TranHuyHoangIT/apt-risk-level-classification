import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL;

console.log('[API Config] Base URL:', BASE_URL);

const api = axios.create({
    baseURL: BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

api.interceptors.request.use(
    (config) => {
        console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`);
        console.log('[API Request Headers]', config.headers);
        console.log('[API Request Data]', config.data);
        const token = localStorage.getItem('token');
        if (token && !['/login', '/register'].includes(config.url)) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        console.error('[API Request Error]', error);
        return Promise.reject(error);
    }
);

api.interceptors.response.use(
    (response) => {
        console.log(`[API Response] ${response.config.method.toUpperCase()} ${response.config.url} - Status: ${response.status}`);
        console.log('[API Response Data]', response.data);
        return response;
    },
    (error) => {
        console.error(`[API Response Error] ${error.config?.method.toUpperCase()} ${error.config?.url} - Status: ${error.response?.status || 'No response'}`);
        console.error('[API Response Error Data]', error.response?.data || error.message);
        return Promise.reject(error);
    }
);

export const login = async (username, password) => {
    try {
        console.log('[Login API] Sending login request', { username, password });
        const res = await api.post('/login', { username, password });
        console.log('[Login API] Login successful', res.data);
        return res.data;
    } catch (error) {
        console.error('[Login API] Login failed', error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi đăng nhập',
            status: error.response?.status,
        };
    }
};

export const register = async (username, email, password) => {
    try {
        const res = await api.post('/register', { username, email, password });
        return res.data;
    } catch (error) {
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi đăng ký',
            status: error.response?.status,
        };
    }
};

// Lấy thông tin hồ sơ
export const getProfile = async () => {
    try {
        console.log('[Profile API] Fetching profile');
        const res = await api.get('/profile');
        console.log('[Profile API] Fetched profile', res.data);
        return res.data;
    } catch (error) {
        console.error('[Profile API] Fetch profile failed', error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi lấy thông tin hồ sơ',
            status: error.response?.status,
        };
    }
};

// Cập nhật thông tin hồ sơ
export const updateProfile = async (profileData) => {
    try {
        console.log('[Profile API] Updating profile', profileData);
        const res = await api.put('/profile', profileData);
        console.log('[Profile API] Updated profile', res.data);
        return res.data;
    } catch (error) {
        console.error('[Profile API] Update profile failed', error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi cập nhật thông tin',
            status: error.response?.status,
        };
    }
};

// Đổi mật khẩu
export const changePassword = async (passwordData) => {
    try {
        console.log('[Profile API] Changing password', passwordData);
        const res = await api.post('/change-password', passwordData);
        console.log('[Profile API] Changed password', res.data);
        return res.data;
    } catch (error) {
        console.error('[Profile API] Change password failed', error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi đổi mật khẩu',
            status: error.response?.status,
        };
    }
};

// // Upload file log
// export const uploadLogs = async (file) => {
//   try {
//     const token = localStorage.getItem('token');
//     console.log('[Upload Debug] Token exists:', !!token);
//     console.log('[Upload Debug] Token preview:', token ? token.substring(0, 20) + '...' : 'No token');

//     const formData = new FormData();
//     formData.append('file', file);
//     const res = await api.post('/upload-logs', formData, {
//       headers: { 'Content-Type': 'multipart/form-data' },
//     });
//     return res.data;
//   } catch (error) {
//     return {
//       error: error.response?.data?.error || 'Đã có lỗi khi upload file log',
//       status: error.response?.status,
//     };
//   }
// };

export const uploadLogs = async (file) => {
  try {
    const token = localStorage.getItem('token');
    
    // Debug decode JWT token
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        console.log('[JWT Debug] Decoded payload:', payload);
        console.log('[JWT Debug] Subject (sub):', payload.sub);
        console.log('[JWT Debug] Expires:', new Date(payload.exp * 1000));
        console.log('[JWT Debug] Current time:', new Date());
      } catch (e) {
        console.error('[JWT Debug] Cannot decode token:', e);
      }
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await api.post('/upload-logs', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  } catch (error) {
    console.error('[Upload Full Error]:', error.response);
    return {
      error: error.response?.data?.error || error.response?.data?.msg || 'Đã có lỗi khi upload file log',
      status: error.response?.status,
    };
  }
};

// Upload file PCAP
export const uploadPcap = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/upload-pcap', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  } catch (error) {
    return {
      error: error.response?.data?.error || 'Đã có lỗi khi upload file PCAP',
      status: error.response?.status,
    };
  }
};

// Lấy lịch sử upload
export const getUploadHistory = async () => {
  try {
    const res = await api.get('/upload-history');
    return res.data;
  } catch (error) {
    return {
      error: error.response?.data?.error || 'Đã có lỗi khi lấy lịch sử upload',
      status: error.response?.status,
    };
  }
};

// Lấy chi tiết upload
export const getUploadDetails = async (uploadId) => {
  try {
    const res = await api.get(`/upload-details/${uploadId}`);
    return res.data;
  } catch (error) {
    return {
      error: error.response?.data?.error || 'Đã có lỗi khi lấy chi tiết upload',
      status: error.response?.status,
    };
  }
};

// Lấy thống kê rủi ro
export const getStageStats = async () => {
  try {
    const res = await api.get('/stage-stats');
    return res.data;
  } catch (error) {
    return {
      error: error.response?.data?.error || 'Đã có lỗi khi lấy thống kê rủi ro',
      status: error.response?.status,
    };
  }
};

// Lấy danh sách uploads
export const getUploads = async () => {
  try {
    const res = await api.get('/Uploads');
    return res.data;
  } catch (error) {
    return {
      error: error.response?.data?.error || 'Đã có lỗi khi lấy danh sách uploads',
      status: error.response?.status,
    };
  }
};

// Lấy danh sách người dùng
export const getUsers = async () => {
    try {
        console.log('[Users API] Fetching users');
        const res = await api.get('/users');
        console.log('[Users API] Fetched users', res.data);
        return res.data;
    } catch (error) {
        console.error('[Users API] Fetch users failed', error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi lấy danh sách người dùng',
            status: error.response?.status,
        };
    }
};

// Lấy thông tin một người dùng
export const getUser = async (userId) => {
    try {
        console.log(`[User API] Fetching user ${userId}`);
        const res = await api.get(`/users/${userId}`);
        console.log(`[User API] Fetched user ${userId}`, res.data);
        return res.data;
    } catch (error) {
        console.error(`[User API] Fetch user ${userId} failed`, error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi lấy thông tin người dùng',
            status: error.response?.status,
        };
    }
};

// Cập nhật thông tin người dùng
export const updateUser = async (userId, userData) => {
    try {
        const { username, email, newPassword } = userData;
        const payload = { username, email };
        if (newPassword) {
            payload.newPassword = newPassword;
        }
        console.log(`[User API] Updating user ${userId}`, payload);
        const res = await api.put(`/users/${userId}`, payload);
        console.log(`[User API] Updated user ${userId}`, res.data);
        return res.data;
    } catch (error) {
        console.error(`[User API] Update user ${userId} failed`, error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi cập nhật thông tin người dùng',
            status: error.response?.status,
        };
    }
};

// Xóa người dùng
export const deleteUser = async (userId) => {
    try {
        console.log(`[User API] Deleting user ${userId}`);
        const res = await api.delete(`/users/${userId}`);
        console.log(`[User API] Deleted user ${userId}`, res.data);
        return res.data;
    } catch (error) {
        console.error(`[User API] Delete user ${userId} failed`, error);
        return {
            error: error.response?.data?.error || 'Đã có lỗi khi xóa người dùng',
            status: error.response?.status,
        };
    }
};

export const simulate = (file, onData, onError) => {
  const formData = new FormData();
  formData.append('file', file);

  console.log('[Simulate API] Sending simulate request', { filename: file.name });

  return new Promise((resolve, reject) => {
    fetch(`${BASE_URL}/simulate`, {
      method: 'POST',
      body: formData,
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function processStream() {
          reader.read().then(({ done, value }) => {
            if (done) {
              console.log('[Simulate API] Stream ended');
              resolve({ close: () => {} });
              return;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete data

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.replace('data: ', ''));
                  console.log('[Simulate API] Received data:', data);
                  if (data.error) {
                    onError(data.error);
                  } else {
                    onData(data);
                  }
                } catch (e) {
                  console.error('[Simulate API] Error parsing data:', e, line);
                  onError(e.message);
                }
              }
            }
            processStream();
          }).catch((err) => {
            console.error('[Simulate API] Stream error:', err);
            onError(err.message);
            reject(err);
          });
        }

        processStream();

        return {
          close: () => {
            reader.cancel();
            console.log('[Simulate API] Stream closed');
          },
        };
      })
      .catch((err) => {
        console.error('[Simulate API] Fetch error:', err);
        onError(err.message);
        reject(err);
      });
  });
};

export const getQueueStatus = async () => {
  console.log('[Queue Status API] Fetching queue status');
  try {
    const response = await api.get('/queue-status');
    console.log('[Queue Status API] Raw response:', response);
    const normalizedResponse = {
      ...response,
      queue_files: Array.isArray(response.queue_files) ? response.queue_files : [],
    };
    console.log('[Queue Status API] Normalized queue status:', normalizedResponse);
    return normalizedResponse;
  } catch (error) {
    console.error('[Queue Status API] Error fetching queue status:', error);
    throw error;
  }
};

export default api;

// import axios from 'axios';

// const BASE_URL = process.env.REACT_APP_API_URL;

// const api = axios.create({
//     baseURL: BASE_URL,
//     headers: {
//         'Content-Type': 'application/json',
//     },
// });

// api.interceptors.request.use(
//     (config) => {
//         console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`);
//         console.log('[API Request Headers]', config.headers);
//         console.log('[API Request Data]', config.data);
//         const token = localStorage.getItem('token');
//         if (token && !['/login', '/register'].includes(config.url)) {
//             config.headers.Authorization = `Bearer ${token}`;
//         }
//         return config;
//     },
//     (error) => {
//         console.error('[API Request Error]', error);
//         return Promise.reject(error);
//     }
// );

// api.interceptors.response.use(
//     (response) => {
//         console.log(`[API Response] ${response.config.method.toUpperCase()} ${response.config.url} - Status: ${response.status}`);
//         console.log('[API Response Data]', response.data);
//         return response;
//     },
//     (error) => {
//         console.error(`[API Response Error] ${error.config?.method.toUpperCase()} ${error.config?.url} - Status: ${error.response?.status || 'No response'}`);
//         console.error('[API Response Error Data]', error.response?.data || error.message);
//         return Promise.reject(error);
//     }
// );

// export const login = async (username, password) => {
//     try {
//         console.log('[Login API] Sending login request', { username, password });
//         const res = await api.post('/login', { username, password });
//         console.log('[Login API] Login successful', res.data);
//         return res.data;
//     } catch (error) {
//         console.error('[Login API] Login failed', error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi đăng nhập',
//             status: error.response?.status,
//         };
//     }
// };

// export const register = async (username, email, password) => {
//     try {
//         const res = await api.post('/register', { username, email, password });
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi đăng ký',
//             status: error.response?.status,
//         };
//     }
// };

// export const getProfile = async () => {
//     try {
//         console.log('[Profile API] Fetching profile');
//         const res = await api.get('/profile');
//         console.log('[Profile API] Fetched profile', res.data);
//         return res.data;
//     } catch (error) {
//         console.error('[Profile API] Fetch profile failed', error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy thông tin hồ sơ',
//             status: error.response?.status,
//         };
//     }
// };

// export const updateProfile = async (profileData) => {
//     try {
//         console.log('[Profile API] Updating profile', profileData);
//         const res = await api.put('/profile', profileData);
//         console.log('[Profile API] Updated profile', res.data);
//         return res.data;
//     } catch (error) {
//         console.error('[Profile API] Update profile failed', error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi cập nhật thông tin',
//             status: error.response?.status,
//         };
//     }
// };

// export const changePassword = async (passwordData) => {
//     try {
//         console.log('[Profile API] Changing password', passwordData);
//         const res = await api.post('/change-password', passwordData);
//         console.log('[Profile API] Changed password', res.data);
//         return res.data;
//     } catch (error) {
//         console.error('[Profile API] Change password failed', error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi đổi mật khẩu',
//             status: error.response?.status,
//         };
//     }
// };

// export const uploadLogs = async (file) => {
//     try {
//         const formData = new FormData();
//         formData.append('file', file);
//         const res = await api.post('/upload-logs', formData, {
//             headers: { 'Content-Type': 'multipart/form-data' },
//         });
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi upload file log',
//             status: error.response?.status,
//         };
//     }
// };

// export const uploadPcap = async (file) => {
//     try {
//         const formData = new FormData();
//         formData.append('file', file);
//         const res = await api.post('/upload-pcap', formData, {
//             headers: { 'Content-Type': 'multipart/form-data' },
//         });
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi upload file PCAP',
//             status: error.response?.status,
//         };
//     }
// };

// export const getUploadHistory = async () => {
//     try {
//         const res = await api.get('/upload-history');
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy lịch sử upload',
//             status: error.response?.status,
//         };
//     }
// };

// export const getUploadDetails = async (uploadId) => {
//     try {
//         const res = await api.get(`/upload-details/${uploadId}`);
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy chi tiết upload',
//             status: error.response?.status,
//         };
//     }
// };

// export const getStageStats = async () => {
//     try {
//         const res = await api.get('/stage-stats');
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy thống kê rủi ro',
//             status: error.response?.status,
//         };
//     }
// };

// export const getUploads = async () => {
//     try {
//         const res = await api.get('/uploads');
//         return res.data;
//     } catch (error) {
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy danh sách uploads',
//             status: error.response?.status,
//         };
//     }
// };

// export const getUsers = async () => {
//     try {
//         console.log('[Users API] Fetching users');
//         const res = await api.get('/users');
//         console.log('[Users API] Fetched users', res.data);
//         return res.data;
//     } catch (error) {
//         console.error('[Users API] Fetch users failed', error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy danh sách người dùng',
//             status: error.response?.status,
//         };
//     }
// };

// export const getUser = async (userId) => {
//     try {
//         console.log(`[User API] Fetching user ${userId}`);
//         const res = await api.get(`/users/${userId}`);
//         console.log(`[User API] Fetched user ${userId}`, res.data);
//         return res.data;
//     } catch (error) {
//         console.error(`[User API] Fetch user ${userId} failed`, error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi lấy thông tin người dùng',
//             status: error.response?.status,
//         };
//     }
// };

// export const updateUser = async (userId, userData) => {
//     try {
//         const { username, email, newPassword } = userData;
//         const payload = { username, email };
//         if (newPassword) {
//             payload.newPassword = newPassword;
//         }
//         console.log(`[User API] Updating user ${userId}`, payload);
//         const res = await api.put(`/users/${userId}`, payload);
//         console.log(`[User API] Updated user ${userId}`, res.data);
//         return res.data;
//     } catch (error) {
//         console.error(`[User API] Update user ${userId} failed`, error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi cập nhật thông tin người dùng',
//             status: error.response?.status,
//         };
//     }
// };

// export const deleteUser = async (userId) => {
//     try {
//         console.log(`[User API] Deleting user ${userId}`);
//         const res = await api.delete(`/users/${userId}`);
//         console.log(`[User API] Deleted user ${userId}`, res.data);
//         return res.data;
//     } catch (error) {
//         console.error(`[User API] Delete user ${userId} failed`, error);
//         return {
//             error: error.response?.data?.error || 'Đã có lỗi khi xóa người dùng',
//             status: error.response?.status,
//         };
//     }
// };

// export const simulate = (file, onData, onError) => {
//     const formData = new FormData();
//     formData.append('file', file);

//     console.log('[Simulate API] Sending simulate request', { filename: file.name });

//     return new Promise((resolve, reject) => {
//         fetch(`${BASE_URL}/simulate`, {
//             method: 'POST',
//             body: formData,
//             headers: {
//                 Authorization: `Bearer ${localStorage.getItem('token')}`,
//             },
//         })
//             .then((response) => {
//                 if (!response.ok) {
//                     throw new Error(`HTTP error! Status: ${response.status}`);
//                 }

//                 const reader = response.body.getReader();
//                 const decoder = new TextDecoder();
//                 let buffer = '';

//                 function processStream() {
//                     reader.read().then(({ done, value }) => {
//                         if (done) {
//                             console.log('[Simulate API] Stream ended');
//                             resolve({ close: () => {} });
//                             return;
//                         }

//                         buffer += decoder.decode(value, { stream: true });
//                         const lines = buffer.split('\n\n');
//                         buffer = lines.pop();

//                         for (const line of lines) {
//                             if (line.startsWith('data: ')) {
//                                 try {
//                                     const data = JSON.parse(line.replace('data: ', ''));
//                                     console.log('[Simulate API] Received data:', data);
//                                     if (data.error) {
//                                         onError(data.error);
//                                     } else {
//                                         onData(data);
//                                     }
//                                 } catch (e) {
//                                     console.error('[Simulate API] Error parsing data:', e, line);
//                                     onError(e.message);
//                                 }
//                             }
//                         }
//                         processStream();
//                     }).catch((err) => {
//                         console.error('[Simulate API] Stream error:', err);
//                         onError(err.message);
//                         reject(err);
//                     });
//                 }

//                 processStream();

//                 return {
//                     close: () => {
//                         reader.cancel();
//                         console.log('[Simulate API] Stream closed');
//                     },
//                 };
//             })
//             .catch((err) => {
//                 console.error('[Simulate API] Fetch error:', err);
//                 onError(err.message);
//                 reject(err);
//             });
//     });
// };

// export const getQueueStatus = async () => {
//     console.log('[Queue Status API] Fetching queue status');
//     try {
//         const response = await api.get('/queue-status');
//         console.log('[Queue Status API] Raw response:', response);
//         const normalizedResponse = {
//             ...response,
//             queue_files: Array.isArray(response.queue_files) ? response.queue_files : [],
//         };
//         console.log('[Queue Status API] Normalized queue status:', normalizedResponse);
//         return normalizedResponse;
//     } catch (error) {
//         console.error('[Queue Status API] Error fetching queue status:', error);
//         throw error;
//     }
// };

// export default api;