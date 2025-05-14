import { useEffect, useState } from 'react';
import { getUploadHistory } from '../services/api';
import { useNavigate } from 'react-router-dom'; 
import {
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Box,
  Divider,
} from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';

export default function UploadHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate(); // Khai báo navigate

  useEffect(() => {
    async function fetchHistory() {
      const res = await getUploadHistory();
      setHistory(res || []);
      setLoading(false);
    }
    fetchHistory();
  }, []);

  const handleSelectUpload = (uploadId) => {
    navigate(`/risk-details/${uploadId}`); 
  };

  return (
    <Card sx={{ mb: 4, boxShadow: 3, borderRadius: 3 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          p: 2,
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          borderTopLeftRadius: 12,
          borderTopRightRadius: 12,
        }}
      >
        <HistoryIcon sx={{ mr: 1 }} />
        <Typography variant="h6" component="div">
          Lịch sử Upload
        </Typography>
      </Box>
      <Divider />
      <CardContent>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 5 }}>
            <CircularProgress />
          </Box>
        ) : history.length === 0 ? (
          <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
            Chưa có upload nào.
          </Typography>
        ) : (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Tổng số lần upload: {history.length}
            </Typography>
            <TableContainer
              component={Paper}
              sx={{
                borderRadius: 2,
                maxHeight: 400,
              }}
            >
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow sx={{ backgroundColor: 'grey.200' }}>
                    <TableCell><strong>Upload ID</strong></TableCell>
                    <TableCell><strong>Filename</strong></TableCell>
                    <TableCell><strong>Upload Time</strong></TableCell>
                    <TableCell align="right"><strong>Tổng số log</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.map((item, index) => (
                    <TableRow
                      key={item.upload_id}
                      hover
                      sx={{
                        cursor: 'pointer',
                        transition: 'transform 0.2s',
                        '&:hover': {
                          backgroundColor: 'action.hover',
                          transform: 'scale(1.01)',
                        },
                        backgroundColor: index % 2 === 0 ? 'grey.50' : 'white',
                      }}
                      onClick={() => handleSelectUpload(item.upload_id)} // Gọi handleSelectUpload
                    >
                      <TableCell>{item.upload_id}</TableCell>
                      <TableCell>{item.filename}</TableCell>
                      <TableCell>{item.upload_time}</TableCell>
                      <TableCell align="right">{item.total_logs}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
      </CardContent>
    </Card>
  );
}
