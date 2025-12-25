import { ChatEn } from '../en/chat';

type I18nKeys = keyof typeof ChatEn;

export interface Resources {
  translation: Record<I18nKeys, string>;
}

export const ChatZh: Resources['translation'] = {
  dialog_list: '对话列表',
  delete_chat: '删除会话',
  delete_chat_confirm: '您确认要删除会话吗？',
  input_tips: '可以问我任何问题，shift + Enter 换行',
  sent: '发送',
  clear_all_caches: '清除所有缓存',
  feedback_tip: '描述一下具体问题或更优的答案',
  thinking: '正在思考中',
  stop_replying: '停止回复',
  erase_memory: '清除记忆',
  copy_success: '复制成功',
  copy_failed: '复制失败',
  copy_nothing: '内容复制为空',
  file_tip: '文件上传后无法更改',
  file_upload_tip: '上传文件到对话（您的模型必须支持多模态输入）',
  chat_online: '在线对话',
  assistant: '平台小助手',
  model_tip: '当前应用暂不支持模型选择',
  temperature_tip: '当前应用暂不支持温度配置',
  max_new_tokens_tip: '当前应用暂不支持max_new_tokens配置',
  extend_tip: '当前应用暂不支持拓展配置',
  cot_title: '思考',
  code_preview: '预览',
  code_preview_full_screen: '全屏',
  code_preview_exit_full_screen: '退出全屏',
  code_preview_code: '代码',
  code_preview_copy: '复制',
  code_preview_already_copied: '已复制',
  code_preview_download: '下载',
  code_preview_run: '运行',
  code_preview_close: '关闭',
  parsing_data: '正在解析理解数据...',
  // Header
  help_center: '帮助中心',
  // ChatHeader
  maybe_you_want_to_ask: '或许你想问：',
  // ToolsBar - Clear all caches dialog
  confirm_clear_all_caches: '确认清除所有缓存？',
  clear_all_caches_content: '此操作将清除以下所有数据：',
  clear_cache_excel_db: 'Excel缓存数据库',
  clear_cache_excel_files: 'Excel数据库文件',
  clear_cache_uploaded_excel: '上传的Excel文件',
  clear_cache_excel_temp_db: 'Excel聊天临时数据库',
  clear_cache_chat_history: '所有会话历史记录',
  clear_cache_file_server: '文件服务器存储',
  clear_cache_model: '模型缓存',
  clear_cache_warning: '此操作不可撤销！',
  confirm_clear: '确认清除',
  clear_cache_success: '所有缓存已清除成功！页面将在3秒后刷新...',
  clear_cache_failed: '清除缓存失败',
  unknown_error: '未知错误',
  // Feedback
  feedback_success: '反馈成功',
  operation_success: '操作成功',
} as const;
