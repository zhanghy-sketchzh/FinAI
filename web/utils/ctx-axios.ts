import axios from 'axios';
import { getApiBaseUrl } from './index';

const api = axios.create({
  // baseURL 将在请求拦截器中动态设置
});

api.defaults.timeout = 10000;

api.interceptors.request.use(request => {
  // 动态设置 baseURL，确保每次请求都使用当前页面的 origin
  request.baseURL = getApiBaseUrl();
  return request;
});

api.interceptors.response.use(
  response => response.data,
  err => Promise.reject(err),
);

export default api;
