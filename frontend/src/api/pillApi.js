import axios from 'axios';

export const uploadPillImage = async (imageBlob) => {
  try {
    const formData = new FormData();
    formData.append('image', imageBlob, 'pill.jpg');

    const response = await axios.post('/api/pills/analyze/', formData);
    return response.data;
  } catch (error) {
    console.error("서버 통신 오류:", error);
    return null;
  }
};