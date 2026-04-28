import React, { useEffect } from 'react';

// 개별 섹션 카드 컴포넌트
const InfoCard = ({ title, content, onClick }) => (
  <div style={cardStyle} onClick={onClick}>
    <h2 style={cardTitleStyle}>{title}</h2>
    <div style={cardContentStyle}>
      {content || '정보가 없습니다.'}
    </div>
  </div>
);

const PillDetail = ({ info, onRetry }) => {
  // 음성 안내 함수 (이전 음성 취소 로직 포함)
  const speak = (text) => {
    window.speechSynthesis.cancel();
    if (!text) return;
    const msg = new SpeechSynthesisUtterance(text);
    msg.lang = 'ko-KR';
    window.speechSynthesis.speak(msg);
  };

  // 페이지 진입 시 안내사항 읽어주기
  useEffect(() => {
    if (info?.effect_text) {
      speak(`분석이 완료되었습니다. 알약의 효능은 ${info.effect_text.slice(0, 40)} 등입니다.`);
    }
    
    // 페이지를 나갈 때 음성 즉시 중단
    return () => {
      window.speechSynthesis.cancel();
    };
  }, [info]);

  // 품목기준코드를 이용한 이미지 경로 생성
  const imageUrl = info?.item_seq
    ? `https://nedrug.mfds.go.kr/pbp/cmn/itemImageDownload/${info.item_seq}`
    : null;

  return (
    <div style={pageStyle}>
      {/* 1. 상단 헤더 바 */}
      <div style={headerBarStyle}>
        <button onClick={onRetry} style={backIconStyle}>〈</button>
        <span style={headerTitleStyle}>의약품 상세정보</span>
        <button onClick={onRetry} style={homeIconStyle}>🏠</button>
      </div>

      <div style={scrollAreaStyle}>
        {/* 2. 알약 이미지 (고정 정사각형 260x260)
        {imageUrl && (
          <div style={imageContainerStyle}>
            <img src={imageUrl} alt="알약" style={squareImgStyle} />
          </div>
        )} */}

        {/* 3. 기본 정보 카드 */}
        <div style={mainInfoCardStyle}>
          <p style={subLabelStyle}>{info?.dl_company}</p>
          <h1 style={itemNameStyle}>{info?.dl_name || `약품번호 ${info?.item_seq}`}</h1>
          <div style={dividerStyle} />
          <div style={pillMetaStyle}>
            {info?.drug_shape && <span style={badgeStyle}>{info.drug_shape}</span>}
            {info?.color_class1 && <span style={badgeStyle}>{info.color_class1}</span>}
            {info?.color_class2 && info.color_class2 !== info.color_class1 && (
              <span style={badgeStyle}>{info.color_class2}</span>
            )}
            {info?.print_front && <span style={badgeStyle}>앞: {info.print_front}</span>}
            {info?.print_back && <span style={badgeStyle}>뒤: {info.print_back}</span>}
          </div>
          {(info?.confidence || info?.method) && (
            <div style={recognitionStyle}>
              {info.method && <p style={recognitionTextStyle}>인식방법: {info.method}</p>}
              {info.confidence && <p style={recognitionTextStyle}>신뢰도: {info.confidence}</p>}
            </div>
          )}
          {info?.tip && <p style={tipStyle}>⚠️ {info.tip}</p>}
          <p style={crawledAtStyle}>품목기준코드: {info?.item_seq}</p>
        </div>

        {/* 4. 상세 내용 카드들 (사용자 데이터 매핑) */}
        <InfoCard 
          title="효능효과" 
          content={info?.effect_text} 
          onClick={() => speak(`효능효과입니다. ${info?.effect_text}`)} 
        />
        <InfoCard 
          title="용법/용량" 
          content={info?.usage_text} 
          onClick={() => speak(`용법용량입니다. ${info?.usage_text}`)} 
        />
        <InfoCard 
          title="사용전 주의사항" 
          content={info?.warning_text} 
          onClick={() => speak(`주의사항입니다. ${info?.warning_text}`)} 
        />
        <InfoCard 
          title="보관방법" 
          content={info?.storage_text} 
          onClick={() => speak(`보관방법입니다. ${info?.storage_text}`)} 
        />

        {/* 5. 공식 출처 링크 카드 */}
        {/* {info?.source_url && (
          <div style={cardStyle}>
            <h2 style={cardTitleStyle}>출처 정보</h2>
            <a href={info.source_url} target="_blank" rel="noreferrer" style={linkStyle}>
              식약처 공식 정보 확인하기 →
            </a>
          </div>
        )} */}

        <button onClick={onRetry} style={retryButtonStyle}>📷 다시 촬영하기</button>
        <div style={{ height: '40px' }} />
      </div>
    </div>
  );
};

/* ── 스타일 정의 ── */

const pageStyle = {
  minHeight: '100vh',
  backgroundColor: '#f5f7fa',
  fontFamily: "'Gowun Dodum', sans-serif",
};

const headerBarStyle = {
  backgroundColor: '#fff',
  padding: '14px 16px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  borderBottom: '1px solid #eee',
  position: 'sticky', top: 0, zIndex: 100,
};

const headerTitleStyle = { fontSize: '18px', fontWeight: 'bold', color: '#333' };
const backIconStyle = { background: 'none', border: 'none', fontSize: '20px', cursor: 'pointer' };
const homeIconStyle = { background: 'none', border: 'none', fontSize: '20px', cursor: 'pointer' };

const scrollAreaStyle = { padding: '16px' };

const imageContainerStyle = {
  backgroundColor: '#fff', borderRadius: '16px', padding: '20px',
  marginBottom: '16px', display: 'flex', justifyContent: 'center',
  boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
};

// ❗ 요청하신 정사각형 260x260 고정 스타일
const squareImgStyle = {
  width: '260px', height: '260px', objectFit: 'cover',
  borderRadius: '12px', backgroundColor: '#f0f0f0',
};

const mainInfoCardStyle = {
  backgroundColor: '#fff', borderRadius: '16px', padding: '24px',
  marginBottom: '16px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
};

const subLabelStyle = { fontSize: '12px', color: '#888', margin: '0 0 4px' };
const itemNameStyle = { fontSize: '22px', fontWeight: 'bold', color: '#1d4ed8', margin: '0 0 12px' };
const dividerStyle = { height: '3px', backgroundColor: '#1d4ed8', marginBottom: '12px', width: '40px' };
const crawledAtStyle = { fontSize: '12px', color: '#999', margin: 0 };

const pillMetaStyle = { display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' };

const badgeStyle = {
  fontSize: '12px', color: '#1d4ed8', backgroundColor: '#eff6ff',
  border: '1px solid #bfdbfe', borderRadius: '20px', padding: '3px 10px',
};

const recognitionStyle = { backgroundColor: '#f8fafc', borderRadius: '8px', padding: '10px', marginBottom: '10px' };
const recognitionTextStyle = { fontSize: '12px', color: '#555', margin: '2px 0' };
const tipStyle = { fontSize: '12px', color: '#b45309', backgroundColor: '#fffbeb', borderRadius: '8px', padding: '8px 12px', marginBottom: '8px' };

const cardStyle = {
  backgroundColor: '#fff', borderRadius: '16px', padding: '20px',
  marginBottom: '16px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
  cursor: 'pointer',
};

const cardTitleStyle = {
  fontSize: '17px', fontWeight: 'bold', color: '#333',
  marginBottom: '12px', borderLeft: '4px solid #1d4ed8', paddingLeft: '10px'
};

const cardContentStyle = { fontSize: '15px', color: '#444', lineHeight: '1.7', whiteSpace: 'pre-wrap' };

const linkStyle = { color: '#1d4ed8', fontSize: '14px', fontWeight: 'bold', textDecoration: 'none' };

const retryButtonStyle = {
  width: '100%', padding: '18px', backgroundColor: '#1d4ed8',
  color: '#fff', border: 'none', borderRadius: '14px',
  fontSize: '17px', fontWeight: 'bold', marginTop: '10px',
};

export default PillDetail;