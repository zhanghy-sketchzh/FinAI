import Image from 'next/image';
import React from 'react';

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
      alt='llm'
    />
  );
}

const ModelSelector: React.FC = () => {
  // 隐藏模型选择器，因为只指定了一个模型
  return null;
  
  // 以下代码已注释，不再使用
  /*
  const { t } = useTranslation();
  const { model, setModel } = useContext(ChatContext);

  const [modelList, setModelList] = useState<string[]>([]);

  useRequest(async () => await apiInterceptors(getUsableModels()), {
    onSuccess: data => {
      const [, res] = data;
      setModelList(res || []);
    },
  });

  if (modelList.length === 0) {
    return null;
  }

  return (
    <div className={styles['cus-selector']}>
      <Select
        value={model}
        placeholder={t('choose_model')}
        className='w-48 h-8 rounded-3xl'
        suffixIcon={<CaretDownOutlined className='text-sm text-[#000000]' />}
        onChange={val => {
          setModel(val);
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
    </div>
  );
  */
};

export default ModelSelector;
