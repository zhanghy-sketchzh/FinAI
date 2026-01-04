/**
 * multi-models selector
 */

import Image from 'next/image';

interface Props {
  onChange?: (model: string) => void;
}

export function renderModelIcon(model?: string, props?: { width: number; height: number }) {
  const { width, height } = props || {};

  if (!model) return null;

  // 统一使用 huggingface.svg 图标
  return (
    <Image
      className='rounded-full border border-gray-200 object-contain bg-white inline-block'
      width={width || 24}
      height={height || 24}
      src='/models/huggingface.svg'
      key='/models/huggingface.svg'
      alt='llm'
    />
  );
}

function ModelSelector(_props: Props) {
  // 隐藏模型选择器，因为只指定了一个模型
  return null;

  // 以下代码已注释，不再使用
  /*
  const { t } = useTranslation();
  const { modelList, model } = useContext(ChatContext);
  if (!modelList || modelList.length <= 0) {
    return null;
  }
  return (
    <Select
      value={model}
      placeholder={t('choose_model')}
      className='w-52'
      onChange={val => {
        onChange?.(val);
      }}
    >
      {modelList.map(item => (
        <Select.Option key={item}>
          <div className='flex items-center'>
            {renderModelIcon(item)}
            <span className='ml-2'>{MODEL_ICON_MAP[item]?.label || item}</span>
          </div>
        </Select.Option>
      ))}
    </Select>
  );
  */
}

export default ModelSelector;
