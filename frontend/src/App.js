import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import PillScanner from './components/PillScanner';
import PillDetail from './components/PillDetail';
import { uploadPillImage } from './api/pillApi';
import mockPill from './mocks/pill.json';

/* ── 스캐너 페이지: / ── */
function ScannerPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleScan = async (blob) => {
    setLoading(true);
    const result = await uploadPillImage(blob);
    setLoading(false);

    if (!result) {
      alert('서버와 통신할 수 없습니다. 잠시 후 다시 시도해주세요.');
      return;
    }

    if (result.status === 'fail') {
      const msg = [result.message, result.tip].filter(Boolean).join('\n\n');
      alert(msg || '알약을 인식하지 못했습니다.');
      return;
    }

    navigate('/detail', { state: { pillData: result } });
  };

  if (loading) {
    return (
      <div style={loadingStyle}>
        <div style={spinnerStyle} />
        <p style={{ marginTop: '20px', color: '#555' }}>알약을 분석하고 있습니다...</p>
      </div>
    );
  }

  return <PillScanner onScanSuccess={handleScan} />;
}

/* ── 상세 페이지: /detail ── */
function DetailPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const pillData = location.state?.pillData;

  if (!pillData) {
    navigate('/');
    return null;
  }

  return <PillDetail info={pillData} onRetry={() => navigate('/')} />;
}

/* ── 목 페이지: /mock ── */
function MockPage() {
  const navigate = useNavigate();
  return <PillDetail info={mockPill} onRetry={() => navigate('/')} />;
}

/* ── 앱 루트 ── */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"       element={<ScannerPage />} />
        <Route path="/detail" element={<DetailPage />} />
        <Route path="/mock"   element={<MockPage />} />
      </Routes>
    </BrowserRouter>
  );
}

const loadingStyle = {
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: '#fff',
};

const spinnerStyle = {
  width: '50px',
  height: '50px',
  border: '5px solid #eee',
  borderTop: '5px solid #3f51b5',
  borderRadius: '50%',
  animation: 'spin 1s linear infinite',
};
