import { apiInterceptors, clearChatHistory, clearAllCaches } from '@/client/api';
import { ChatContentContext } from '@/pages/chat';
import { ClearOutlined, DeleteOutlined, LoadingOutlined, PauseCircleOutlined, RedoOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { Modal, Spin, Tooltip, message } from 'antd';
import classNames from 'classnames';
import Image from 'next/image';
import React, { useContext, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { parseResourceValue, transformFileUrl } from '@/utils';

import MaxNewTokens from './MaxNewTokens';
import ModelSwitcher from './ModelSwitcher';
import Resource from './Resource';
import Temperature from './Temperature';

interface ToolsConfig {
  icon: React.ReactNode;
  can_use: boolean;
  key: string;
  tip?: string;
  onClick?: () => void;
}

const ToolsBar: React.FC<{
  ctrl: AbortController;
  onLoadingChange?: (loading: boolean) => void;
}> = ({ ctrl, onLoadingChange }) => {
  const { t } = useTranslation();

  const {
    history,
    scrollRef,
    canAbort,
    replyLoading,
    currentDialogue,
    appInfo,
    temperatureValue,
    maxNewTokensValue,
    resourceValue,
    setTemperatureValue,
    setMaxNewTokensValue,
    refreshHistory,
    setCanAbort,
    setReplyLoading,
    handleChat,
  } = useContext(ChatContentContext);

  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [clsLoading, setClsLoading] = useState<boolean>(false);
  const [clearAllLoading, setClearAllLoading] = useState<boolean>(false);

  // 通知父组件 loading 状态变化
  React.useEffect(() => {
    onLoadingChange?.(loading);
  }, [loading, onLoadingChange]);

  // 左边工具栏动态可用key
  const paramKey: string[] = useMemo(() => {
    return appInfo.param_need?.map(i => i.type) || [];
  }, [appInfo.param_need]);

  const rightToolsConfig: ToolsConfig[] = useMemo(() => {
    return [
      {
        tip: t('stop_replying'),
        icon: <PauseCircleOutlined className={classNames({ 'text-[#0c75fc]': canAbort })} />,
        can_use: canAbort,
        key: 'abort',
        onClick: () => {
          if (!canAbort) {
            return;
          }
          ctrl.abort();
          setTimeout(() => {
            setCanAbort(false);
            setReplyLoading(false);
          }, 100);
        },
      },
      {
        tip: t('clear_all_caches'),
        icon: clearAllLoading ? (
          <Spin spinning={clearAllLoading} indicator={<LoadingOutlined style={{ fontSize: 20 }} />} />
        ) : (
          <DeleteOutlined />
        ),
        can_use: !replyLoading,
        key: 'clear_all_caches',
        onClick: async () => {
          if (clearAllLoading) {
            return;
          }
          
          Modal.confirm({
            title: '确认清除所有缓存？',
            content: (
              <div>
                <p>此操作将清除以下所有数据：</p>
                <ul style={{ paddingLeft: '20px', margin: '10px 0' }}>
                  <li>Excel缓存数据库</li>
                  <li>Excel数据库文件</li>
                  <li>上传的Excel文件</li>
                  <li>Excel聊天临时数据库</li>
                  <li>所有会话历史记录</li>
                  <li>文件服务器存储</li>
                  <li>模型缓存</li>
                </ul>
                <p style={{ color: 'red', fontWeight: 'bold' }}>此操作不可撤销！</p>
              </div>
            ),
            okText: '确认清除',
            cancelText: '取消',
            okType: 'danger',
            onOk: async () => {
              setClearAllLoading(true);
              try {
                await apiInterceptors(clearAllCaches());
                message.success('所有缓存已清除成功！页面将在3秒后刷新...');
                
                // 3秒后刷新页面
                setTimeout(() => {
                  window.location.reload();
                }, 3000);
              } catch (error: any) {
                message.error(`清除缓存失败: ${error?.message || '未知错误'}`);
              } finally {
                setClearAllLoading(false);
              }
            },
          });
        },
      },
      {
        tip: t('erase_memory'),
        icon: clsLoading ? (
          <Spin spinning={clsLoading} indicator={<LoadingOutlined style={{ fontSize: 20 }} />} />
        ) : (
          <ClearOutlined />
        ),
        can_use: history.length > 0,
        key: 'clear',
        onClick: async () => {
          if (clsLoading) {
            return;
          }
          setClsLoading(true);
          await apiInterceptors(clearChatHistory(currentDialogue.conv_uid)).finally(async () => {
            await refreshHistory();
            setClsLoading(false);
          });
        },
      },
    ];
  }, [
    t,
    canAbort,
    replyLoading,
    history,
    clsLoading,
    ctrl,
    setCanAbort,
    setReplyLoading,
    handleChat,
    appInfo.app_code,
    paramKey,
    temperatureValue,
    resourceValue,
    currentDialogue.select_param,
    currentDialogue.conv_uid,
    scrollRef,
    refreshHistory,
  ]);

  const returnTools = (config: ToolsConfig[]) => {
    return (
      <>
        {config.map(item => (
          <Tooltip key={item.key} title={item.tip} arrow={false} placement='bottom'>
            <div
              className={`flex w-8 h-8 items-center justify-center rounded-md hover:bg-[rgb(221,221,221,0.6)] text-lg ${
                item.can_use ? 'cursor-pointer' : 'opacity-30 cursor-not-allowed'
              }`}
              onClick={() => {
                item.onClick?.();
              }}
            >
              {item.icon}
            </div>
          </Tooltip>
        ))}
      </>
    );
  };

  const fileName = useMemo(() => {
    try {
      // First try to get file_name from resourceValue
      if (resourceValue) {
        if (typeof resourceValue === 'string') {
          return JSON.parse(resourceValue).file_name || '';
        } else {
          return resourceValue.file_name || '';
        }
      }
      // Fall back to currentDialogue.select_param if resourceValue doesn't have file_name
      return JSON.parse(currentDialogue.select_param).file_name || '';
    } catch {
      return '';
    }
  }, [resourceValue, currentDialogue.select_param]);

  const ResourceItemsDisplay = () => {
    const resources = parseResourceValue(resourceValue) || parseResourceValue(currentDialogue.select_param) || [];

    if (resources.length === 0) return null;

    return (
      <div className='group/item flex flex-wrap gap-2 mt-2'>
        {resources.map((item, index) => {
          // Handle image type
          if (item.type === 'image_url' && item.image_url?.url) {
            const fileName = item.image_url.fileName;
            const previewUrl = transformFileUrl(item.image_url.url);
            return (
              <div
                key={`img-${index}`}
                className='flex flex-col border border-[#e3e4e6] dark:border-[rgba(255,255,255,0.6)] rounded-lg p-2'
              >
                {/* Add image preview */}
                <div className='w-32 h-32 mb-2 overflow-hidden flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded'>
                  <img src={previewUrl} alt={fileName || 'Preview'} className='max-w-full max-h-full object-contain' />
                </div>
                <div className='flex items-center'>
                  <span className='text-sm text-[#1c2533] dark:text-white line-clamp-1'>{fileName}</span>
                </div>
              </div>
            );
          }
          // Handle file type
          else if (item.type === 'file_url' && item.file_url?.url) {
            const fileName = item.file_url.file_name;

            return (
              <div
                key={`file-${index}`}
                className='flex items-center justify-between border border-[#e3e4e6] dark:border-[rgba(255,255,255,0.6)] rounded-lg p-2'
              >
                <div className='flex items-center'>
                  <Image src={`/icons/chat/excel.png`} width={20} height={20} alt='file-icon' className='mr-2' />
                  <span className='text-sm text-[#1c2533] dark:text-white line-clamp-1'>{fileName}</span>
                </div>
              </div>
            );
          }

          return null;
        })}
      </div>
    );
  };

  return (
    <div className='flex flex-col  mb-2'>
      <div className='flex items-center justify-between h-full w-full'>
        <div className='flex gap-3 text-lg'>
          <ModelSwitcher />
          <Resource fileList={fileList} setFileList={setFileList} setLoading={setLoading} fileName={fileName} />
          <Temperature temperatureValue={temperatureValue} setTemperatureValue={setTemperatureValue} />
          <MaxNewTokens maxNewTokensValue={maxNewTokensValue} setMaxNewTokensValue={setMaxNewTokensValue} />
        </div>
        <div className='flex gap-1'>{returnTools(rightToolsConfig)}</div>
      </div>
      <ResourceItemsDisplay />
      {loading && (
        <div className='flex items-center gap-2 mt-2'>
          <Spin spinning={loading} indicator={<LoadingOutlined style={{ fontSize: 16 }} spin />} />
          <span className='text-sm text-gray-500'>{t('parsing_data')}</span>
        </div>
      )}
    </div>
  );
};

export default ToolsBar;
