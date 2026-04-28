// src/mocks/mockData.js

export const mockSuccess = {
  id: "432",
  item_seq: "197000037",
  effect_text: "이 약은 육체피로, 임신ㆍ수유기, 병중ㆍ병후의 체력 저하 시, 노년기의 비타민 B1, B2, B6, E, C의 보급과 신경통, 근육통, 관절통(요통, 어깨결림 등) 증상의 완화, 각기, 눈의 피로에 사용합니다.",
  usage_text: "성인은 1회 1정씩, 1일 2회 복용합니다.",
  warning_text: "이 약에 과민증 환자, 만 3개월 미만의 젖먹이는 이 약을 복용하지 마십시오. 위부불쾌감, 발진, 구역 등이 나타나는 경우 복용을 중지하고 의사 또는 약사와 상의하십시오.",
  storage_text: "습기와 빛을 피해 실온에서 보관하십시오. 어린이의 손이 닿지 않는 곳에 보관하십시오.",
  source_url: "https://nedrug.mfds.go.kr/pbp/cmn/xml/drb/197000037/NN",
  crawl_status: "SUCCESS_FULL",
  crawled_at: "2026-04-13 14:16:39",
};

// 서버 오류 또는 알약 미인식 시 null 반환
export const mockError = null;