import { ChatContext } from '@/app/chat-context';
import { ChatContentContext } from '@/pages/chat';
import { SettingOutlined } from '@ant-design/icons';
import { Select, Tooltip } from 'antd';
import React, { memo, useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import ModelIcon from '../content/ModelIcon';

/**
 * 格式化模型名称，使其更清晰易读
 * 例如：
 * - "Qwen/Qwen3-Coder-30B-A3B-Instruct" -> "Qwen3-Coder-30B"
 * - "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" -> "DeepSeek-R1-32B"
 * - "deepseek-r1:70b" -> "DeepSeek-R1-70B"
 */
const formatModelName = (modelName: string): string => {
  if (!modelName) return modelName;

  // 移除前缀（如 "Qwen/", "deepseek-ai/"）
  let name = modelName.includes('/') ? modelName.split('/').pop() || modelName : modelName;

  // 处理 ollama 格式 (model:size)
  if (name.includes(':')) {
    const [model, size] = name.split(':');
    // 格式化模型名称为首字母大写
    const formattedModel = model
      .split('-')
      .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join('-');
    return `${formattedModel}-${size.toUpperCase()}`;
  }

  // 移除常见后缀词
  const suffixesToRemove = ['Instruct', 'Chat', 'Base', 'Distill', 'Qwen', 'Llama'];
  suffixesToRemove.forEach(suffix => {
    // 移除末尾的后缀（如 -Instruct, -Chat）
    const suffixPattern = new RegExp(`-${suffix}$`, 'i');
    name = name.replace(suffixPattern, '');
    // 移除中间的 Distill-Qwen 等组合
    const middlePattern = new RegExp(`-Distill-(?:Qwen|Llama)`, 'i');
    name = name.replace(middlePattern, '');
  });

  // 提取并格式化参数大小（如 30B, 70B, 32B）
  const sizeMatch = name.match(/(\d+[BbMm])/);
  if (sizeMatch) {
    const size = sizeMatch[1].toUpperCase();
    // 移除所有大小标记，只保留最后一个
    name = name
      .replace(/[-_]?\d+[BbMm][-_]?/g, '-')
      .replace(/-+/g, '-')
      .replace(/-$/, '');
    // 移除 A3B 这类中间标记
    name = name.replace(/-A\d+B/i, '');
    name = `${name}-${size}`;
  }

  // 清理多余的连字符
  name = name.replace(/-+/g, '-').replace(/^-|-$/g, '');

  return name;
};

const ModelSwitcher: React.FC = () => {
  const { modelList } = useContext(ChatContext);
  const { appInfo, modelValue, setModelValue } = useContext(ChatContentContext);

  const { t } = useTranslation();

  // 左边工具栏动态可用key
  const paramKey: string[] = useMemo(() => {
    return appInfo.param_need?.map(i => i.type) || [];
  }, [appInfo.param_need]);

  if (!paramKey.includes('model')) {
    return (
      <Tooltip title={t('model_tip')}>
        <div className='flex w-8 h-8 items-center justify-center rounded-md hover:bg-[rgb(221,221,221,0.6)]'>
          <SettingOutlined className='text-xl cursor-not-allowed opacity-30' />
        </div>
      </Tooltip>
    );
  }

  return (
    <Select
      value={modelValue}
      placeholder={t('choose_model')}
      className='h-8 rounded-3xl'
      onChange={val => {
        setModelValue(val);
      }}
      popupMatchSelectWidth={280}
      optionLabelProp='label'
    >
      {modelList.map(item => (
        <Select.Option key={item} value={item} label={formatModelName(item)}>
          <Tooltip title={item} placement='right'>
            <div className='flex items-center'>
              <ModelIcon model={item} />
              <span className='ml-2'>{formatModelName(item)}</span>
            </div>
          </Tooltip>
        </Select.Option>
      ))}
    </Select>
  );
};

export default memo(ModelSwitcher);
