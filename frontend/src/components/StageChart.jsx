import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function StageChart({ data }) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-4">
      <h3 className="text-lg font-semibold mb-4">Phân bố giai đoạn tấn công</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey="stage_label" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
