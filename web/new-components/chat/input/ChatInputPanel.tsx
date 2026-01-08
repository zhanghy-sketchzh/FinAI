import { ChatContentContext } from '@/pages/chat';
import { DownOutlined, LoadingOutlined, ReloadOutlined, UpOutlined } from '@ant-design/icons';
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
    scrollRef,
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
  const [displayedQuestions, setDisplayedQuestions] = useState<string[]>([]); // 当前显示的问题
  const [allQuestions, setAllQuestions] = useState<string[]>([]); // 所有9个问题
  const prevConvUidRef = useRef<string>(''); // 用于跟踪会话ID的变化
  const lastProcessedMessageRef = useRef<string>(''); // 用于跟踪已处理的消息，避免重复处理
  const prevLanguageRef = useRef<string>(i18n.language); // 用于跟踪语言变化

  const submitCountRef = useRef(0);

  const paramKey: string[] = useMemo(() => {
    return appInfo.param_need?.map(i => i.type) || [];
  }, [appInfo.param_need]);

  // 判断是否有文件上传（用于 chat_excel 场景）
  const hasFileUploaded = useMemo(() => {
    if (scene !== 'chat_excel') return true; // 非 chat_excel 场景始终显示输入框
    
    try {
      // 检查 resourceValue
      if (resourceValue) {
        if (typeof resourceValue === 'string') {
          const parsed = JSON.parse(resourceValue);
          if (parsed.file_name || parsed.original_filename) return true;
        } else {
          if (resourceValue.file_name || resourceValue.original_filename) return true;
        }
      }
      
      // 检查 currentDialogue.select_param
      if (currentDialogue?.select_param && currentDialogue?.conv_uid === chatId) {
        const parsed = JSON.parse(currentDialogue.select_param);
        if (parsed.file_name || parsed.original_filename) return true;
      }
      
      // 检查 parseResourceValue
      const selectParam = currentDialogue?.select_param && currentDialogue?.conv_uid === chatId
        ? currentDialogue.select_param
        : null;
      const resources = parseResourceValue(resourceValue) || parseResourceValue(selectParam) || [];
      const fileResources = resources.filter(item => item.type === 'file_url' && item.file_url?.url);
      if (fileResources.length > 0) return true;
      
      return false;
    } catch {
      return false;
    }
  }, [scene, resourceValue, currentDialogue?.select_param, currentDialogue?.conv_uid, chatId]);

  // 当会话ID变化时，清除动态推荐问题
  useEffect(() => {
    const currentConvUid = currentDialogue?.conv_uid || '';
    if (currentConvUid && prevConvUidRef.current && currentConvUid !== prevConvUidRef.current) {
      setDynamicSuggestedQuestions([]);
      setAllQuestions([]);
      setDisplayedQuestions([]);
      prevConvUidRef.current = currentConvUid;
    } else if (!currentConvUid && prevConvUidRef.current) {
      setDynamicSuggestedQuestions([]);
      setAllQuestions([]);
      setDisplayedQuestions([]);
      prevConvUidRef.current = '';
    } else if (currentConvUid && !prevConvUidRef.current) {
      prevConvUidRef.current = currentConvUid;
    }
  }, [currentDialogue?.conv_uid]);

  // 当语言切换时，清除动态推荐问题，使用对应语言的初始推荐问题
  useEffect(() => {
    if (i18n.language !== prevLanguageRef.current) {
      setDynamicSuggestedQuestions([]);
      setAllQuestions([]);
      setDisplayedQuestions([]);
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
            setAllQuestions(questions);
            // 从9个问题中随机选择2个简单问题 + 1个开放式问题
            const selectedQuestions = selectRandomQuestions(questions);
            setDynamicSuggestedQuestions(selectedQuestions);
            setDisplayedQuestions(selectedQuestions);
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
                setAllQuestions(questions);
                // 从9个问题中随机选择2个简单问题 + 1个开放式问题
                const selectedQuestions = selectRandomQuestions(questions);
                setDynamicSuggestedQuestions(selectedQuestions);
                setDisplayedQuestions(selectedQuestions);
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
      
      return Array.isArray(questions) ? questions : [];
    } catch (error) {
      return [];
    }
  }, [scene, resourceValue, currentDialogue, chatId, i18n.language]);

  // 从9个问题中随机选择2个简单问题 + 1个开放式问题的函数
  const selectRandomQuestions = (questions: string[]): string[] => {
    if (questions.length <= 3) {
      return questions;
    }
    // 假设前6个是简单问题，后3个是开放式问题
    const simpleQuestions = questions.slice(0, 6);
    const openQuestions = questions.slice(6, 9);
    
    // 随机选择2个简单问题
    const shuffledSimple = [...simpleQuestions].sort(() => Math.random() - 0.5);
    const selectedSimple = shuffledSimple.slice(0, 2);
    
    // 随机选择1个开放式问题
    const shuffledOpen = [...openQuestions].sort(() => Math.random() - 0.5);
    const selectedOpen = shuffledOpen.slice(0, 1);
    
    return [...selectedSimple, ...selectedOpen];
  };

  // 刷新推荐问题
  const handleRefreshQuestions = () => {
    const questionsToUse = allQuestions.length > 0 ? allQuestions : (initialSuggestedQuestions.length > 0 ? initialSuggestedQuestions : []);
    if (questionsToUse.length > 0) {
      const selectedQuestions = selectRandomQuestions(questionsToUse);
      setDisplayedQuestions(selectedQuestions);
    }
  };

  // 当初始推荐问题变化时，更新allQuestions和displayedQuestions
  useEffect(() => {
    if (initialSuggestedQuestions.length > 0 && allQuestions.length === 0 && dynamicSuggestedQuestions.length === 0) {
      setAllQuestions(initialSuggestedQuestions);
      const selectedQuestions = selectRandomQuestions(initialSuggestedQuestions);
      setDisplayedQuestions(selectedQuestions);
    }
  }, [initialSuggestedQuestions, allQuestions.length, dynamicSuggestedQuestions.length]);

  // 优先使用动态推荐问题，如果没有则使用初始推荐问题
  const suggestedQuestions = useMemo(() => {
    if (dynamicSuggestedQuestions.length > 0) {
      return displayedQuestions.length > 0 ? displayedQuestions : dynamicSuggestedQuestions;
    }
    // 对于初始推荐问题，使用已选择的显示问题
    if (initialSuggestedQuestions.length > 0) {
      return displayedQuestions.length > 0 ? displayedQuestions : selectRandomQuestions(initialSuggestedQuestions);
    }
    return [];
  }, [dynamicSuggestedQuestions, initialSuggestedQuestions, displayedQuestions]);

  // 处理推荐问题点击
  const handleSuggestedQuestionClick = async (question: string) => {
    if (!question.trim() || replyLoading || fileUploading) {
      return;
    }
    await onSubmit(question);
    // 滚动到底部
    setTimeout(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current?.scrollHeight,
        behavior: 'smooth',
      });
    }, 100);
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
        <div className='mb-4'>
          <div className='flex items-start gap-3'>
            {/* 左侧标签 */}
            <div className='flex items-center gap-1.5 flex-shrink-0 py-1'>
              <span className='text-[13px] text-slate-500 dark:text-slate-400 whitespace-nowrap'>
                {t('maybe_you_want_to_ask') || '或许你想问'}
              </span>
            </div>

            {/* 右侧问题列表 */}
            <div className='flex-1 flex flex-wrap items-center gap-2'>
              {suggestedQuestions.map((question, index) => (
                <div
                  key={index}
                  className='inline-flex items-center px-3 py-1.5 bg-white/70 dark:bg-[rgba(255,255,255,0.08)] border border-slate-200/80 dark:border-[rgba(255,255,255,0.15)] rounded-lg cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-300 dark:hover:border-blue-500/40 transition-all duration-200 group/item'
                  onClick={() => handleSuggestedQuestionClick(question)}
                >
                  <span className='text-[13px] text-slate-600 dark:text-slate-300 group-hover/item:text-blue-600 dark:group-hover/item:text-blue-400 transition-colors leading-5'>
                    {question}
                  </span>
                </div>
              ))}
              {/* 刷新按钮 */}
              <div
                onClick={handleRefreshQuestions}
                className='inline-flex items-center gap-1 px-2.5 py-1.5 text-slate-400 dark:text-slate-500 hover:text-blue-500 dark:hover:text-blue-400 cursor-pointer transition-colors'
                title={t('refresh_questions_tip')}
              >
                <ReloadOutlined className='text-xs' />
                <span className='text-xs'>{t('refresh_questions')}</span>
              </div>
            </div>
          </div>
        </div>
      )}
      <div
        className={`flex flex-1 flex-col bg-white dark:bg-[rgba(255,255,255,0.16)] px-5 py-4 pt-2 rounded-xl relative border-t border-b border-l border-r dark:border-[rgba(255,255,255,0.6)] ${
          isFocus ? 'border-[#0c75fc]' : ''
        }`}
        id='input-panel'
      >
        <ToolsBar ctrl={ctrl} onLoadingChange={setFileUploading} />
        {hasFileUploaded && (
          <>
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
          </>
        )}
      </div>
    </div>
  );
};

export default forwardRef(ChatInputPanel);
