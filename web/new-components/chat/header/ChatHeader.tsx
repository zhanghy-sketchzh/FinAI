import { ChatContext } from '@/app/chat-context';
import { apiInterceptors, collectApp, newDialogue, unCollectApp } from '@/client/api';
import { ChatContentContext } from '@/pages/chat';
import { STORAGE_INIT_MESSAGE_KET } from '@/utils';
import { LoadingOutlined, PlusOutlined, StarFilled, StarOutlined } from '@ant-design/icons';
import { Spin, Tag, Typography } from 'antd';
import { useRouter } from 'next/router';
import React, { useCallback, useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useRequest } from 'ahooks';
import AppDefaultIcon from '../../common/AppDefaultIcon';

const tagColors = ['magenta', 'orange', 'geekblue', 'purple', 'cyan', 'green'];

const ChatHeader: React.FC<{ isScrollToTop: boolean }> = ({ isScrollToTop }) => {
  const {
    appInfo,
    refreshAppInfo,
    handleChat,
    scrollRef,
    temperatureValue,
    resourceValue,
    currentDialogue,
    refreshDialogList,
    setResourceValue,
    setHistory,
  } = useContext(ChatContentContext);
  const { model, setCurrentDialogInfo } = useContext(ChatContext);
  const router = useRouter();

  const { t } = useTranslation();

  // 新建会话
  const handleNewChat = useCallback(async () => {
    // 清除所有相关状态，确保创建全新的对话
    setResourceValue?.(null);
    setHistory?.([]);
    // 清除初始化消息
    localStorage.removeItem(STORAGE_INIT_MESSAGE_KET);

    const [, res] = await apiInterceptors(newDialogue({ chat_mode: 'chat_excel', model }));
    if (res) {
      setCurrentDialogInfo?.({
        chat_scene: 'chat_excel',
        app_code: '',
      });
      localStorage.setItem(
        'cur_dialog_info',
        JSON.stringify({
          chat_scene: 'chat_excel',
          app_code: '',
        }),
      );
      // 使用 replace 而不是 push，避免浏览器历史记录
      router.replace(`/chat?scene=chat_excel&id=${res.conv_uid}${model ? `&model=${model}` : ''}`);
      refreshDialogList?.();
    }
  }, [model, router, setCurrentDialogInfo, refreshDialogList, setResourceValue, setHistory]);

  const appScene = useMemo(() => {
    return appInfo?.team_context?.chat_scene || 'chat_agent';
  }, [appInfo]);

  // 获取应用名称和描述，如果是 chat_excel 则使用翻译
  const displayAppName = useMemo(() => {
    if (appScene === 'chat_excel') {
      return t('chat_excel_app_name');
    }
    return appInfo?.app_name;
  }, [appScene, appInfo?.app_name, t]);

  const displayAppDescribe = useMemo(() => {
    if (appScene === 'chat_excel') {
      return t('chat_excel_app_describe');
    }
    return appInfo?.app_describe;
  }, [appScene, appInfo?.app_describe, t]);

  // 应用收藏状态
  const isCollected = useMemo(() => {
    return appInfo?.is_collected === 'true';
  }, [appInfo]);

  const { run: operate, loading } = useRequest(
    async () => {
      const [error] = await apiInterceptors(
        isCollected ? unCollectApp({ app_code: appInfo.app_code }) : collectApp({ app_code: appInfo.app_code }),
      );
      if (error) {
        return;
      }
      return await refreshAppInfo();
    },
    {
      manual: true,
    },
  );

  const paramKey: string[] = useMemo(() => {
    return appInfo.param_need?.map(i => i.type) || [];
  }, [appInfo.param_need]);

  if (!Object.keys(appInfo).length) {
    return null;
  }

  // 正常header
  const headerContent = () => {
    return (
      <header className='flex items-center justify-between w-5/6 h-full px-6  bg-[#ffffff99] border dark:bg-[rgba(255,255,255,0.1)] dark:border-[rgba(255,255,255,0.1)] rounded-2xl mx-auto transition-all duration-400 ease-in-out relative'>
        <div className='flex items-center'>
          <div className='flex w-12 h-12 justify-center items-center rounded-xl mr-4 bg-white'>
            <AppDefaultIcon scene={appScene} width={16} height={16} />
          </div>
          <div className='flex flex-col flex-1'>
            <div className='flex items-center text-base text-[#1c2533] dark:text-[rgba(255,255,255,0.85)] font-semibold gap-2'>
              <span>{displayAppName}</span>
            </div>
            <Typography.Text
              className='text-sm text-[#525964] dark:text-[rgba(255,255,255,0.65)] leading-6'
              ellipsis={{
                tooltip: true,
              }}
            >
              {displayAppDescribe}
            </Typography.Text>
          </div>
        </div>
        <div className='flex items-center gap-3'>
          {/* 新建会话 */}
          <div
            onClick={handleNewChat}
            className='flex items-center gap-2 px-4 h-10 bg-[#ffffff99] dark:bg-[rgba(255,255,255,0.2)] border border-white dark:border-[rgba(255,255,255,0.2)] rounded-full cursor-pointer hover:bg-[#f0f0f0] dark:hover:bg-[rgba(255,255,255,0.3)]'
          >
            <PlusOutlined style={{ fontSize: 16 }} />
            <span className='text-sm font-medium'>{t('new_chat')}</span>
          </div>
          {/* 收藏 */}
          <div
            onClick={async () => {
              await operate();
            }}
            className='flex items-center justify-center w-10 h-10 bg-[#ffffff99] dark:bg-[rgba(255,255,255,0.2)] border border-white dark:border-[rgba(255,255,255,0.2)] rounded-[50%] cursor-pointer'
          >
            {loading ? (
              <Spin spinning={loading} indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
            ) : (
              <>
                {isCollected ? (
                  <StarFilled style={{ fontSize: 18 }} className='text-yellow-400 cursor-pointer' />
                ) : (
                  <StarOutlined style={{ fontSize: 18, cursor: 'pointer' }} />
                )}
              </>
            )}
          </div>
        </div>
        {!!appInfo?.recommend_questions?.length && (
          <div className='absolute  bottom-[-40px] left-0'>
            <span className='text-sm text-[#525964] dark:text-[rgba(255,255,255,0.65)] leading-6'>
              {t('maybe_you_want_to_ask')}
            </span>
            {appInfo.recommend_questions.map((item, index) => (
              <Tag
                key={item.id}
                color={tagColors[index]}
                className='text-xs p-1 px-2 cursor-pointer'
                onClick={async () => {
                  handleChat(item?.question || '', {
                    app_code: appInfo.app_code,
                    ...(paramKey.includes('temperature') && { temperature: temperatureValue }),
                    ...(paramKey.includes('resource') && {
                      select_param:
                        typeof resourceValue === 'string'
                          ? resourceValue
                          : JSON.stringify(resourceValue) || currentDialogue.select_param,
                    }),
                  });
                  setTimeout(() => {
                    scrollRef.current?.scrollTo({
                      top: scrollRef.current?.scrollHeight,
                      behavior: 'smooth',
                    });
                  }, 0);
                }}
              >
                {item.question}
              </Tag>
            ))}
          </div>
        )}
      </header>
    );
  };
  // 吸顶header
  const topHeaderContent = () => {
    return (
      <header className='flex items-center justify-between w-full h-14 bg-[#f0f7ff] dark:bg-[#1a2332] px-8 transition-all duration-500 ease-in-out'>
        <div className='flex items-center'>
          <div className='flex items-center justify-center w-8 h-8 rounded-lg mr-2 bg-white'>
            <AppDefaultIcon scene={appScene} />
          </div>
          <div className='flex items-center text-base text-[#1c2533] dark:text-[rgba(255,255,255,0.85)] font-semibold gap-2'>
            <span>{displayAppName}</span>
          </div>
        </div>
        <div className='flex items-center gap-4'>
          {/* 新建会话 */}
          <div
            onClick={handleNewChat}
            className='flex items-center gap-1.5 px-3 h-8 rounded-full cursor-pointer hover:bg-[#f0f0f0] dark:hover:bg-[rgba(255,255,255,0.2)]'
          >
            <PlusOutlined style={{ fontSize: 14 }} />
            <span className='text-sm font-medium'>{t('new_chat')}</span>
          </div>
          {/* 收藏 */}
          <div
            onClick={async () => {
              await operate();
            }}
            className='cursor-pointer'
          >
            {loading ? (
              <Spin spinning={loading} indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
            ) : (
              <>
                {isCollected ? (
                  <StarFilled style={{ fontSize: 18 }} className='text-yellow-400 cursor-pointer' />
                ) : (
                  <StarOutlined style={{ fontSize: 18, cursor: 'pointer' }} />
                )}
              </>
            )}
          </div>
        </div>
      </header>
    );
  };

  return (
    <div
      className={`h-20 mt-6 ${
        appInfo?.recommend_questions && appInfo?.recommend_questions?.length > 0 ? 'mb-6' : ''
      } sticky top-0 bg-transparent z-30 transition-all duration-400 ease-in-out`}
    >
      {isScrollToTop ? topHeaderContent() : headerContent()}
    </div>
  );
};

export default ChatHeader;
