import axios from 'axios';
import { getApiBaseUrl } from './index';

const api = axios.create({
  baseURL: getApiBaseUrl(),
});

api.defaults.timeout = 10000;

api.interceptors.response.use(
  response => response.data,
  err => Promise.reject(err),
);

export default api;
