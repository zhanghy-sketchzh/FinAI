import axios from 'axios';
import { getApiBaseUrl } from './index';

const api = axios.create({
  // baseURL 将在请求拦截器中动态设置
});

api.defaults.timeout = 10000;

api.interceptors.request.use(request => {
  // 动态设置 baseURL，确保使用当前页面的 origin
  if (!request.baseURL) {
    request.baseURL = getApiBaseUrl();
  }
  return request;
});

api.interceptors.response.use(
  response => response.data,
  err => Promise.reject(err),
);

export default api;
