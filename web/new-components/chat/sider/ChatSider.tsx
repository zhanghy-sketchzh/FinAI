import { ChatContext } from '@/app/chat-context';
import { apiInterceptors, delDialogue, newDialogue } from '@/client/api';
import { ChatContentContext } from '@/pages/chat';
import { DarkSvg, SunnySvg } from '@/components/icons';
import UserBar from '@/new-components/layout/UserBar';
import { IChatDialogueSchema } from '@/types/chat';
import { STORAGE_INIT_MESSAGE_KET, STORAGE_LANG_KEY, STORAGE_THEME_KEY } from '@/utils/constants/index';
import Icon, {
  CaretLeftOutlined,
  CaretRightOutlined,
  DeleteOutlined,
  GlobalOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { Flex, Layout, Modal, Popover, Spin, Tooltip, Typography } from 'antd';
import moment from 'moment';
import 'moment/locale/zh-cn';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import React, { useCallback, useContext, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppDefaultIcon from '../../common/AppDefaultIcon';

const { Sider } = Layout;

const zeroWidthTriggerDefaultStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 16,
  height: 48,
  position: 'absolute',
  top: '50%',
  transform: 'translateY(-50%)',
  border: '1px solid #d6d8da',
  borderRadius: 8,
  right: -8,
};

/**
 * 会话项
 */
const MenuItem: React.FC<{
  item: any;
  refresh?: any;
  order: React.MutableRefObject<number>;
  historyLoading?: boolean;
}> = ({ item, refresh, historyLoading }) => {
  const { t } = useTranslation();
  const router = useRouter();
  const searchParams = useSearchParams();
  const chatId = searchParams?.get('id') ?? '';
  const scene = searchParams?.get('scene') ?? '';

  const { setCurrentDialogInfo } = useContext(ChatContext);

  // 当前活跃会话
  const active = useMemo(() => {
    if (item.default) {
      return item.default && !chatId && !scene;
    } else {
      return item.conv_uid === chatId && item.chat_mode === scene;
    }
  }, [chatId, scene, item]);

  // 删除会话
  const handleDelChat = () => {
    Modal.confirm({
      title: t('delete_chat'),
      content: t('delete_chat_confirm'),
      centered: true,
      onOk: async () => {
        const [err] = await apiInterceptors(delDialogue(item.conv_uid));
        if (err) {
          return;
        }
        await refresh?.();
        if (item.conv_uid === chatId) {
          router.push(`/chat`);
        }
      },
    });
  };

  return (
    <Flex
      align='center'
      className={`group/item w-full h-12 p-3 rounded-lg  hover:bg-white dark:hover:bg-theme-dark cursor-pointer mb-2 relative ${
        active ? 'bg-white dark:bg-theme-dark bg-opacity-100' : ''
      }`}
      onClick={() => {
        if (historyLoading) {
          return;
        }
        !item.default &&
          setCurrentDialogInfo?.({
            chat_scene: item.chat_mode,
            app_code: item.app_code,
          });
        localStorage.setItem(
          'cur_dialog_info',
          JSON.stringify({
            chat_scene: item.chat_mode,
            app_code: item.app_code,
          }),
        );
        router.push(item.default ? '/chat' : `?scene=${item.chat_mode}&id=${item.conv_uid}`);
      }}
    >
      <Tooltip title={item.chat_mode}>
        <div className='flex items-center justify-center w-8 h-8 rounded-lg mr-3 bg-white'>{item.icon}</div>
      </Tooltip>
      <div className='flex flex-1 line-clamp-1'>
        <Typography.Text
          ellipsis={{
            tooltip: true,
          }}
        >
          {item.label}
        </Typography.Text>
      </div>
      {!item.default && (
        <div className='flex gap-1 ml-1'>
          <div
            className='group-hover/item:opacity-100 cursor-pointer opacity-0'
            onClick={e => {
              e.stopPropagation();
              handleDelChat();
            }}
          >
            <DeleteOutlined style={{ fontSize: 16 }} />
          </div>
        </div>
      )}
      <div
        className={` w-1 rounded-sm bg-[#0c75fc] absolute top-1/2 left-0 -translate-y-1/2 transition-all duration-500 ease-in-out ${
          active ? 'h-5' : 'w-0 h-0'
        }`}
      />
    </Flex>
  );
};

const ChatSider: React.FC<{
  dialogueList: any;
  refresh: () => void;
  historyLoading: boolean;
  listLoading: boolean;
  order: React.MutableRefObject<number>;
}> = ({ dialogueList = [], refresh, historyLoading, listLoading, order }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const scene = searchParams?.get('scene') ?? '';
  const { t, i18n } = useTranslation();
  const { mode, setMode, model, setCurrentDialogInfo } = useContext(ChatContext);
  const { setResourceValue, setHistory } = useContext(ChatContentContext);
  const [collapsed, setCollapsed] = useState<boolean>(scene === 'chat_dashboard');

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
      refresh?.();
    }
  }, [model, router, setCurrentDialogInfo, refresh, setResourceValue, setHistory]);

  // 切换主题
  const handleToggleTheme = useCallback(() => {
    const theme = mode === 'light' ? 'dark' : 'light';
    setMode(theme);
    localStorage.setItem(STORAGE_THEME_KEY, theme);
  }, [mode, setMode]);

  // 切换语言
  const handleChangeLang = useCallback(() => {
    const language = i18n.language === 'en' ? 'zh' : 'en';
    i18n.changeLanguage(language);
    if (language === 'zh') moment.locale('zh-cn');
    if (language === 'en') moment.locale('en');
    localStorage.setItem(STORAGE_LANG_KEY, language);
  }, [i18n]);

  // 展开或收起列表按钮样式
  const triggerStyle: React.CSSProperties = useMemo(() => {
    if (collapsed) {
      return {
        ...zeroWidthTriggerDefaultStyle,
        right: -16,
        borderRadius: '0px 8px 8px 0',
        borderLeft: '1px solid #d5e5f6',
      };
    }
    return {
      ...zeroWidthTriggerDefaultStyle,
      borderLeft: '1px solid #d6d8da',
    };
  }, [collapsed]);

  // 会话列表配置项
  const items: MenuProps['items'] = useMemo(() => {
    const list = dialogueList[1] || [];
    if (list?.length > 0) {
      return list.map((item: IChatDialogueSchema) => ({
        ...item,
        label: item.user_input || item.select_param,
        key: item.conv_uid,
        icon: <AppDefaultIcon scene={item.chat_mode} />,
        default: false,
      }));
    }
    return [];
  }, [dialogueList]);

  return (
    <Sider
      className='bg-[#ffffff80] border-r border-[#d5e5f6] dark:bg-[#ffffff29] dark:border-[#ffffff66]'
      theme={mode}
      width={280}
      collapsible={true}
      collapsed={collapsed}
      collapsedWidth={0}
      trigger={collapsed ? <CaretRightOutlined className='text-base' /> : <CaretLeftOutlined className='text-base' />}
      zeroWidthTriggerStyle={triggerStyle}
      onCollapse={collapsed => setCollapsed(collapsed)}
    >
      <div className='flex flex-col h-full w-full bg-transparent'>
        {/* Logo */}
        <div className='px-4 pt-4 pb-4'>
          <Link href='/' className='flex items-center justify-center'>
            <Image src='/finai_logo.png' alt='FinAI' width={180} height={45} />
          </Link>
        </div>

        {/* 对话列表标题 */}
        <div className='px-4 pt-2'>
          <div className='flex items-center justify-between mb-4'>
            <div className='text-base font-semibold text-[#1c2533] dark:text-[rgba(255,255,255,0.85)] line-clamp-1'>
              {t('dialog_list')}
            </div>
            <Tooltip title={t('new_chat')}>
              <div
                onClick={handleNewChat}
                className='flex items-center justify-center w-7 h-7 rounded-md bg-[#0c75fc] hover:bg-[#0a5fd4] text-white cursor-pointer'
              >
                <PlusOutlined style={{ fontSize: 14 }} />
              </div>
            </Tooltip>
          </div>
        </div>

        {/* 对话列表 */}
        <Flex flex={1} vertical={true} className='overflow-y-auto px-4'>
          <Spin spinning={listLoading} className='mt-2'>
            {!!items?.length &&
              items.map(item => (
                <MenuItem key={item?.key} item={item} refresh={refresh} historyLoading={historyLoading} order={order} />
              ))}
          </Spin>
        </Flex>

        {/* 底部设置区域 */}
        <div className='px-4 py-2 border-t border-dashed border-gray-200 dark:border-gray-700'>
          <div className='flex items-center justify-end gap-1'>
            {/* 主题切换 */}
            <Popover content={t('Theme')}>
              <div
                className='flex items-center justify-center w-8 h-8 rounded-md hover:bg-[#F1F5F9] dark:hover:bg-theme-dark cursor-pointer text-lg'
                onClick={handleToggleTheme}
              >
                {mode === 'dark' ? <Icon component={DarkSvg} /> : <Icon component={SunnySvg} />}
              </div>
            </Popover>
            {/* 语言切换 */}
            <Popover content={t('language')}>
              <div
                className='flex items-center justify-center w-8 h-8 rounded-md hover:bg-[#F1F5F9] dark:hover:bg-theme-dark cursor-pointer text-lg'
                onClick={handleChangeLang}
              >
                <GlobalOutlined />
              </div>
            </Popover>
          </div>
        </div>
      </div>
    </Sider>
  );
};

export default ChatSider;
