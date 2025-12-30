import { ChatContentContext } from '@/pages/chat';
import { DownOutlined, LoadingOutlined, UpOutlined } from '@ant-design/icons';
import { Button, Input, Spin, Tag } from 'antd';
import classNames from 'classnames';
import { useSearchParams } from 'next/navigation';
import React, { forwardRef, useContext, useImperativeHandle, useMemo, useRef, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { UserChatContent } from '@/types/chat';
import { parseResourceValue } from '@/utils';
import ToolsBar from './ToolsBar';

const tagColors = ['geekblue', 'purple', 'cyan', 'green', 'orange'];

const ChatInputPanel: React.ForwardRefRenderFunction<any, { ctrl: AbortController }> = ({ ctrl }, ref) => {
  const { t, i18n } = useTranslation();
  const {
    replyLoading,
    handleChat,
    appInfo,
    currentDialogue,
    temperatureValue,
    maxNewTokensValue,
    resourceValue,
    setResourceValue,
    refreshDialogList,
    history,
  } = useContext(ChatContentContext);

  const searchParams = useSearchParams();
  const scene = searchParams?.get('scene') ?? '';
  const select_param = searchParams?.get('select_param') ?? '';
  const chatId = searchParams?.get('id') ?? '';

  const [userInput, setUserInput] = useState<string>('');
  const [isFocus, setIsFocus] = useState<boolean>(false);
  const [isZhInput, setIsZhInput] = useState<boolean>(false);
  const [fileUploading, setFileUploading] = useState<boolean>(false);
  const [dynamicSuggestedQuestions, setDynamicSuggestedQuestions] = useState<string[]>([]);
  const [isQuestionsCollapsed, setIsQuestionsCollapsed] = useState<boolean>(false);
  const prevConvUidRef = useRef<string>(''); // 用于跟踪会话ID的变化
  const lastProcessedMessageRef = useRef<string>(''); // 用于跟踪已处理的消息，避免重复处理
  const prevLanguageRef = useRef<string>(i18n.language); // 用于跟踪语言变化

  const submitCountRef = useRef(0);

  const paramKey: string[] = useMemo(() => {
    return appInfo.param_need?.map(i => i.type) || [];
  }, [appInfo.param_need]);

  // 当会话ID变化时，清除动态推荐问题
  useEffect(() => {
    const currentConvUid = currentDialogue?.conv_uid || '';
    if (currentConvUid && prevConvUidRef.current && currentConvUid !== prevConvUidRef.current) {
      setDynamicSuggestedQuestions([]);
      prevConvUidRef.current = currentConvUid;
    } else if (!currentConvUid && prevConvUidRef.current) {
      setDynamicSuggestedQuestions([]);
      prevConvUidRef.current = '';
    } else if (currentConvUid && !prevConvUidRef.current) {
      prevConvUidRef.current = currentConvUid;
    }
  }, [currentDialogue?.conv_uid]);

  // 当语言切换时，清除动态推荐问题，使用对应语言的初始推荐问题
  useEffect(() => {
    if (i18n.language !== prevLanguageRef.current) {
      setDynamicSuggestedQuestions([]);
      prevLanguageRef.current = i18n.language;
    }
  }, [i18n.language]);

  // 从最新的消息响应中提取动态推荐问题
  useEffect(() => {
    if (scene !== 'chat_excel' || !history || history.length === 0) {
      return;
    }

    // 找到最新的view消息（按order排序，找到order最大的view消息）
    const viewMessages = history.filter(msg => msg.role === 'view' && msg.context);
    if (viewMessages.length === 0) {
      return;
    }
    
    const latestViewMessage = viewMessages.reduce((latest, current) => {
      return (current.order || 0) > (latest.order || 0) ? current : latest;
    });

    if (!latestViewMessage || !latestViewMessage.context) {
      return;
    }

    const context = latestViewMessage.context;
    const contextStr = typeof context === 'string' ? context : JSON.stringify(context);
    
    const messageHash = `${latestViewMessage.order}_${contextStr.slice(-500)}`;
    if (messageHash === lastProcessedMessageRef.current && contextStr.includes('<!--SUGGESTED_QUESTIONS:')) {
      return;
    }

    const startMarker = '<!--SUGGESTED_QUESTIONS:';
    const endMarker = '-->';
    const startIndex = contextStr.indexOf(startMarker);
    
    if (startIndex !== -1) {
      const jsonStartIndex = startIndex + startMarker.length;
      const remainingStr = contextStr.substring(jsonStartIndex);
      const endIndex = remainingStr.lastIndexOf(endMarker);
      
      if (endIndex > 0) {
        const jsonStr = remainingStr.substring(0, endIndex);
        try {
          const questionsData = JSON.parse(jsonStr);
          const questions = questionsData.suggested_questions || [];
          if (Array.isArray(questions) && questions.length > 0) {
            setDynamicSuggestedQuestions(questions.slice(0, 4));
            lastProcessedMessageRef.current = messageHash;
            return;
          }
        } catch (error) {
          // 如果解析失败，尝试使用正则表达式作为后备方案
          const suggestedQuestionsMatch = contextStr.match(/<!--SUGGESTED_QUESTIONS:(\{[\s\S]*?\})-->/);
          if (suggestedQuestionsMatch) {
            try {
              const questionsData = JSON.parse(suggestedQuestionsMatch[1]);
              const questions = questionsData.suggested_questions || [];
              if (Array.isArray(questions) && questions.length > 0) {
                setDynamicSuggestedQuestions(questions.slice(0, 4));
                lastProcessedMessageRef.current = messageHash;
                return;
              }
            } catch (e) {
              // 解析失败，忽略
            }
          }
        }
      }
    }

    lastProcessedMessageRef.current = messageHash;
  }, [history, scene]);

  // 从select_param或resourceValue中提取初始推荐问题
  const initialSuggestedQuestions = useMemo(() => {
    if (scene !== 'chat_excel') {
      return [];
    }

    try {
      // 优先从resourceValue中获取
      let selectParamStr = '';
      if (resourceValue) {
        if (typeof resourceValue === 'string') {
          selectParamStr = resourceValue;
        } else {
          selectParamStr = JSON.stringify(resourceValue);
        }
      } else if (currentDialogue?.select_param && currentDialogue?.conv_uid === chatId) {
        // 只有当 currentDialogue.conv_uid 匹配当前 chatId 时才使用
        // 这样可以确保新会话时不会显示上一个会话的推荐问题
        selectParamStr = currentDialogue.select_param;
      }

      if (!selectParamStr) {
        return [];
      }

      // 尝试解析JSON
      let selectParam: any = {};
      try {
        selectParam = typeof selectParamStr === 'string' ? JSON.parse(selectParamStr) : selectParamStr;
      } catch {
        return [];
      }

      // 从data_schema_json中提取推荐问题
      const dataSchemaJson = selectParam.data_schema_json;
      if (!dataSchemaJson) {
        return [];
      }

      let schema: any = {};
      try {
        schema = typeof dataSchemaJson === 'string' ? JSON.parse(dataSchemaJson) : dataSchemaJson;
      } catch {
        return [];
      }

      // 根据用户语言选择推荐问题版本
      const isEnglish = i18n.language === 'en';
      let questions: string[] = [];
      
      if (isEnglish) {
        // 优先使用英文版本
        questions = schema.suggested_questions_en || [];
      } else {
        // 使用中文版本
        questions = schema.suggested_questions_zh || [];
      }
      
      // 兼容旧格式：如果新格式没有，尝试使用旧格式 suggested_questions（作为中文版本）
      if (!questions || questions.length === 0) {
        questions = schema.suggested_questions || [];
      }
      
      return Array.isArray(questions) ? questions.slice(0, 4) : [];
    } catch (error) {
      return [];
    }
  }, [scene, resourceValue, currentDialogue, chatId, i18n.language]);

  // 优先使用动态推荐问题，如果没有则使用初始推荐问题
  const suggestedQuestions = useMemo(() => {
    return dynamicSuggestedQuestions.length > 0 ? dynamicSuggestedQuestions : initialSuggestedQuestions;
  }, [dynamicSuggestedQuestions, initialSuggestedQuestions]);

  // 处理推荐问题点击
  const handleSuggestedQuestionClick = async (question: string) => {
    if (!question.trim() || replyLoading || fileUploading) {
      return;
    }
    await onSubmit(question);
  };

  const onSubmit = async (inputText?: string) => {
    submitCountRef.current++;
    // Remove immediate scroll to avoid conflict with ChatContentContainer's auto-scroll
    // ChatContentContainer will handle scrolling when new content is added
    const textToSubmit = inputText || userInput;
    setUserInput('');
    const resources = parseResourceValue(resourceValue);
    // Clear the resourceValue if it not empty
    let newUserInput: UserChatContent;
    if (resources.length > 0) {
      if (scene !== 'chat_excel') {
        // Chat Excel scene does not need to clear the resourceValue
        // We need to find a better way to handle this
        setResourceValue(null);
      }
      const messages = [...resources];
      messages.push({
        type: 'text',
        text: textToSubmit,
      });
      newUserInput = {
        role: 'user',
        content: messages,
      };
    } else {
      newUserInput = textToSubmit;
    }

    const params = {
      app_code: appInfo.app_code || '',
      ...(paramKey.includes('temperature') && { temperature: temperatureValue }),
      ...(paramKey.includes('max_new_tokens') && { max_new_tokens: maxNewTokensValue }),
      select_param,
      ...(paramKey.includes('resource') && {
        select_param:
          typeof resourceValue === 'string'
            ? resourceValue
            : JSON.stringify(resourceValue) || currentDialogue.select_param,
      }),
    };

    await handleChat(newUserInput, params);

    // 如果应用进来第一次对话，刷新对话列表
    if (submitCountRef.current === 1) {
      await refreshDialogList();
    }
  };

  // expose setUserInput to parent via ref
  useImperativeHandle(ref, () => ({
    setUserInput,
  }));

  return (
    <div className='flex flex-col w-5/6 mx-auto pt-4 pb-6 bg-transparent'>
      {/* 推荐问题气泡展示 */}
      {suggestedQuestions.length > 0 && (
        <div className='mb-4 flex flex-col gap-2'>
          <div
            className='flex items-center gap-2 cursor-pointer select-none hover:opacity-80 transition-opacity'
            onClick={() => setIsQuestionsCollapsed(!isQuestionsCollapsed)}
          >
            <span className='text-sm text-[#525964] dark:text-[rgba(255,255,255,0.65)] leading-6'>
              {t('maybe_you_want_to_ask') || '您可能想问：'}
            </span>
            {isQuestionsCollapsed ? (
              <DownOutlined className='text-xs text-[#525964] dark:text-[rgba(255,255,255,0.65)] transition-transform' />
            ) : (
              <UpOutlined className='text-xs text-[#525964] dark:text-[rgba(255,255,255,0.65)] transition-transform' />
            )}
          </div>
          {!isQuestionsCollapsed && (
            <div className='flex flex-wrap gap-2'>
              {suggestedQuestions.map((question, index) => (
                <Tag
                  key={index}
                  color={tagColors[index % tagColors.length]}
                  className='text-xs px-3 py-1 cursor-pointer hover:opacity-80 transition-opacity rounded-full'
                  onClick={() => handleSuggestedQuestionClick(question)}
                >
                  {question}
                </Tag>
              ))}
            </div>
          )}
        </div>
      )}
      <div
        className={`flex flex-1 flex-col bg-white dark:bg-[rgba(255,255,255,0.16)] px-5 py-4 pt-2 rounded-xl relative border-t border-b border-l border-r dark:border-[rgba(255,255,255,0.6)] ${
          isFocus ? 'border-[#0c75fc]' : ''
        }`}
        id='input-panel'
      >
        <ToolsBar ctrl={ctrl} onLoadingChange={setFileUploading} />
        <Input.TextArea
          placeholder={t('input_tips')}
          className='w-full h-20 resize-none border-0 p-0 focus:shadow-none dark:bg-transparent'
          value={userInput}
          onKeyDown={e => {
            if (e.key === 'Enter') {
              if (e.shiftKey) {
                return;
              }
              if (isZhInput) {
                return;
              }
              e.preventDefault();
              if (!userInput.trim() || replyLoading || fileUploading) {
                return;
              }
              onSubmit();
            }
          }}
          onChange={e => {
            setUserInput(e.target.value);
          }}
          onFocus={() => {
            setIsFocus(true);
          }}
          onBlur={() => setIsFocus(false)}
          onCompositionStart={() => setIsZhInput(true)}
          onCompositionEnd={() => setIsZhInput(false)}
        />
        <Button
          type='primary'
          className={classNames(
            'flex items-center justify-center w-14 h-8 rounded-lg text-sm absolute right-4 bottom-3 bg-button-gradient border-0',
            {
              'cursor-not-allowed': !userInput.trim() || fileUploading,
              'opacity-40': fileUploading,
            },
          )}
          disabled={fileUploading}
          onClick={() => {
            if (replyLoading || !userInput.trim() || fileUploading) {
              return;
            }
            onSubmit();
          }}
        >
          {replyLoading ? (
            <Spin spinning={replyLoading} indicator={<LoadingOutlined className='text-white' />} />
          ) : (
            t('sent')
          )}
        </Button>
      </div>
    </div>
  );
};

export default forwardRef(ChatInputPanel);
