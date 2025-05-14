import { useParams } from 'react-router-dom'; 
import UploadDetails from '../components/UploadDetails';

export default function RiskDetailsPage() {
  const { uploadId } = useParams();

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold mb-4">Chi tiết bản ghi</h1>
      <UploadDetails uploadId={uploadId} />
    </div>
  );
}


