import React, { useRef, useEffect, useState } from 'react';

const PillScanner = ({ onScanSuccess }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const shapes = [
    { id: 'circle', name: '원형', radius: '50%'},
    { id: 'oval', name: '타원형', radius: '100px / 150px', height:'240px'},
    { id: 'oblong', name: '장방형', radius: '80px', width: '220px', height: '320px'},
    { id: 'triangle', name: '삼각형', radius: '0'},
  ];

  const [showIntro, setShowIntro] = useState(true); // ← 추가
  const [shapeIndex, setShapeIndex] = useState(0);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedBlob, setSelectedBlob] = useState(null);
  const [imageSource, setImageSource] = useState(null);

  const speak = (text) => {
    //새로운 음성을 시작하기 전에 이전 음성을 취소
    window.speechSynthesis.cancel();

    if (!text) return;

    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = 'ko-KR';
    window.speechSynthesis.speak(msg);
  };

  useEffect(() => {
    if (showIntro) {
      speak('알약을 촬영하기 전에 안내사항을 확인해주세요.');
    }

    else if (!showIntro && !previewUrl){
        startCamera();
        speak(`${shapes[shapeIndex].name} 가이드입니다. 알약을 점선 안에 배치해 주세요.`);
    }
    
    return () => {
        window.speechSynthesis.cancel();
    };    
  }, [showIntro, previewUrl, shapeIndex]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      speak("카메라 권한을 확인해주세요.");
    }
  };

  const nextShape = () => {
    const nextIdx = (shapeIndex + 1) % shapes.length;
    setShapeIndex(nextIdx);
    speak(`${shapes[nextIdx].name} 모양으로 변경되었습니다.`);
  };

  const prevShape = () => {
    const prevIdx = (shapeIndex - 1 + shapes.length) % shapes.length;
    setShapeIndex(prevIdx);
    speak(`${shapes[prevIdx].name} 모양으로 변경되었습니다.`);
  };

  const capture = () => {
    const context = canvasRef.current.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, 640, 480);
    canvasRef.current.toBlob((blob) => {
      setPreviewUrl(URL.createObjectURL(blob));
      setSelectedBlob(blob);
      setImageSource('camera');
      speak("사진이 촬영되었습니다. 이미지를 확인하거나 다시 찍어주세요.");
    }, 'image/jpeg');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
      setSelectedBlob(file);
      setImageSource('file');
      speak("이미지가 선택되었습니다. 사진을 확인해 주세요.");
    }
  };

  const handleConfirm = () => {
    speak("알약 분석을 시작합니다.");
    onScanSuccess(selectedBlob);
  };

  const handleRetry = () => {
    setPreviewUrl(null);
    setSelectedBlob(null);
    speak("다시 촬영 모드로 돌아갑니다.");
  };

  // ─── 인트로 화면 ───────────────────────────────────────────
  if (showIntro) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f5f5f5', padding: '20px' }}>
        <div style={{ width: '100%', maxWidth: '360px', background: '#fff', borderRadius: '28px', padding: '28px 24px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>

          <div style={{ background: '#eff6ff', color: '#1d4ed8', borderRadius: '20px', padding: '8px 20px', fontSize: '14px', fontWeight: '500' }}>
            알약 촬영 시
          </div>

          <p style={{ textAlign: 'center', fontSize: '16px', lineHeight: '1.9', color: '#222', margin: 0 }}>
            1. 알약이 <span style={{ color: '#2563eb', fontWeight: '600' }}>네모칸 중앙</span>에 오도록<br />
            2. <span style={{ color: '#2563eb', fontWeight: '600' }}>밝은</span> 조명 아래서<br />
            3. 대비되는 <span style={{ color: '#2563eb', fontWeight: '600' }}>깔끔한</span> 배경에서
          </p>

          {/* 나쁜 예시 */}
          <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '12px', background: '#f9f9f9', borderRadius: '14px', padding: '12px' }}>
            <div style={{ width: '72px', height: '60px', borderRadius: '10px', background: '#c8c8c8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px', flexShrink: 0 }}>💊✏️</div>
            <span style={{ fontSize: '13px', color: '#888', flex: 1 }}>복잡한 배경</span>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', border: '3px solid #e24b4a', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <svg width="14" height="14" viewBox="0 0 14 14">
                <line x1="2" y1="2" x2="12" y2="12" stroke="#e24b4a" strokeWidth="2.5" strokeLinecap="round" />
                <line x1="12" y1="2" x2="2" y2="12" stroke="#e24b4a" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
          </div>

          {/* 좋은 예시 */}
          <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '12px', background: '#f9f9f9', borderRadius: '14px', padding: '12px' }}>
            <div style={{ width: '72px', height: '60px', borderRadius: '10px', background: '#1e2a38', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px', flexShrink: 0 }}>💊</div>
            <span style={{ fontSize: '13px', color: '#888', flex: 1 }}>깔끔한 배경</span>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', border: '3px solid #2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <svg width="16" height="16" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="6" stroke="#2563eb" strokeWidth="2.5" fill="none" />
              </svg>
            </div>
          </div>

          <button
            onClick={() => {
              speak('촬영을 시작합니다.');
              setShowIntro(false);
            }}
            style={{ width: '100%', padding: '16px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '16px', fontSize: '17px', fontWeight: '600', cursor: 'pointer', marginTop: '4px' }}
          >
            알약 촬영하기
          </button>

        </div>
      </div>
    );
  }

  // ─── 기존 스캐너 화면 ──────────────────────────────────────
  return (
    <div className="scanner-container" style={{ position: 'relative', height: '100vh', background: '#000', overflow: 'hidden' }}>
    {previewUrl ? (
  <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', background: '#fff', padding: '30px 20px' }}>
    
    {/* 상단 배지 */}
    <div style={{ background: '#eff6ff', color: '#1d4ed8', borderRadius: '20px', padding: '8px 24px', fontSize: '15px', fontWeight: '500', marginBottom: '24px' }}>
      사진 확인
    </div>

    {/* 안내 문구 */}
    <p style={{ textAlign: 'center', fontSize: '15px', lineHeight: '1.9', color: '#222', margin: '0 0 24px' }}>
      1. 알약이 <span style={{ color: '#2563eb', fontWeight: '600' }}>네모칸 중앙</span>에 오도록<br />
      2. <span style={{ color: '#2563eb', fontWeight: '600' }}>밝은</span> 조명 아래서<br />
      3. 대비되는 <span style={{ color: '#2563eb', fontWeight: '600' }}>깔끔한</span> 배경에서
    </p>

    {/* 촬영 확인 문구 */}
    <p style={{ fontSize: '15px', color: '#444', marginBottom: '14px' }}>알약 촬영이 잘 되었나요?</p>

    {/* 사진 */}
    <img
      src={previewUrl}
      alt="Preview"
      style = {squareImgStyle} 
    //   style={{ width: '85%', maxHeight: '280px', objectFit: 'contain', borderRadius: '16px', marginBottom: '32px', boxShadow: '0 4px 15px rgba(0,0,0,0.12)' }}
    />

    {/* 사진 확인 화면에서의 버튼 영역 */}
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '85%' }}>
    {imageSource === 'file' ? (
    // 갤러리에서 가져온 경우
    <button 
      onClick={() => {
        handleRetry(); // 기존 상태 초기화
        fileInputRef.current.click(); // 바로 파일 선택창 열기
      }} 
      style={secondaryBtnStyle}
    >
      📁 이미지 다시 선택하기
    </button>
  ) : (
    // 카메라로 촬영한 경우
    <button onClick={handleRetry} style={secondaryBtnStyle}>
      📷 사진 다시 촬영하기
    </button>
  )}
  
    <button onClick={handleConfirm} style={primaryBtnStyle}>
    알약 검색하기
    </button>
  </div>
</div>
) : (
        <>
          <video ref={videoRef} autoPlay playsInline style={videoStyle} />
          <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
          <div style={topTextStyle}>
            <p>약품 모양 점선 안에 약을 배치하고 촬영해 주세요.</p>
            <p style={{ fontSize: '14px', color: '#ccc' }}>(꽉 안 채우셔도 돼요)</p>
          </div>
          <div style={centerControlsStyle}>
            <button onClick={prevShape} style={arrowBtnStyle}>◀</button>
            <div style={{ width: '240px', height: '240px', border: '4px dashed #fff', borderRadius: shapes[shapeIndex].radius, transition: 'all 0.3s ease' }}></div>
            <button onClick={nextShape} style={arrowBtnStyle}>▶</button>
          </div>
          <div style={bottomTextStyle}>
            <p>다른 모양의 약품을 촬영하려면<br />화면을 옆으로 넘겨주세요.</p>
          </div>
          <div style={bottomBarStyle}>
            <button onClick={() => fileInputRef.current.click()} style={iconBtnStyle}>📁</button>
            <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" style={{ display: 'none' }} />
            <button onClick={capture} style={captureBtnStyle}>알약 촬영</button>
          </div>
        </>
      )}
    </div>
  );
};

const videoStyle = { width: '100%', height: '100%', objectFit: 'cover', opacity: 0.6 };
const previewImgStyle = { maxWidth: '80%', maxHeight: '60vh', width: 'auto', height: 'auto', objectFit: 'contain', borderRadius: '20px', marginBottom: '30px', boxShadow: '0 4px 15px rgba(0,0,0,0.15)' };
const topTextStyle = { position: 'absolute', top: '8%', width: '100%', textAlign: 'center', fontSize: '18px', color: '#fff', padding: '0 20px', zIndex: 10 };
const centerControlsStyle = { position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: '20px', zIndex: 10 };
const arrowBtnStyle = { background: 'none', border: 'none', color: '#fff', fontSize: '30px', cursor: 'pointer' };
const bottomTextStyle = { position: 'absolute', bottom: '140px', width: '100%', textAlign: 'center', color: '#ddd', fontSize: '15px', zIndex: 10 };
const bottomBarStyle = { position: 'absolute', bottom: '40px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '30px', zIndex: 10 };
const captureBtnStyle = { padding: '15px 40px', fontSize: '18px', borderRadius: '30px', backgroundColor: '#fff', border: 'none', fontWeight: 'bold' };
const iconBtnStyle = { width: '50px', height: '50px', borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.3)', border: 'none', fontSize: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center' };
const primaryBtnStyle = { width: '100%', padding: '15px 0', backgroundColor: '#1d4ed8', color: '#fff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: 'bold' };
const secondaryBtnStyle = { width: '100%', padding: '15px 0', backgroundColor: '#eff6ff', color: '#1d4ed8', border: 'none', borderRadius: '12px', fontSize: '16px' };
const squareImgStyle = { width: '300px', height: '300px', objectFit: 'cover', objectPosition: 'center', borderRadius: '16px', backgroundColor: '#000', boxShadow: '0 4px 15px rgba(0,0,0,0.12)', marginBottom: '32px' };

export default PillScanner;