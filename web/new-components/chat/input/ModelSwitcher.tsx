import React, { memo } from 'react';

const ModelSwitcher: React.FC = () => {
  // 隐藏模型选择器，因为只指定了一个模型
  return null;
  
  // 以下代码已注释，不再使用
  /*
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
  */
};

export default memo(ModelSwitcher);
