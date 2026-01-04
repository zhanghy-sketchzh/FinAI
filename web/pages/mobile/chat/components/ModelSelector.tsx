import React from 'react';

const ModelSelector: React.FC = () => {
  // 隐藏模型选择器，因为只指定了一个模型
  return null;

  // 以下代码已注释，不再使用
  /*
  const { modelList } = useContext(ChatContext);
  const { model, setModel } = useContext(MobileChatContext);

  const items: MenuProps['items'] = useMemo(() => {
    if (modelList.length > 0) {
      return modelList.map(item => {
        return {
          label: (
            <div
              className='flex items-center gap-2'
              onClick={() => {
                setModel(item);
              }}
            >
              <ModelIcon width={14} height={14} model={item} />
              <span className='text-xs'>{item}</span>
            </div>
          ),
          key: item,
        };
      });
    }
    return [];
  }, [modelList, setModel]);

  return (
    <Dropdown
      menu={{
        items,
      }}
      placement='top'
      trigger={['click']}
    >
      <Popover content={model}>
        <div className='flex items-center gap-1 border rounded-xl bg-white dark:bg-black p-2 flex-shrink-0'>
          <ModelIcon width={16} height={16} model={model} />
          <span
            className='text-xs font-medium line-clamp-1'
            style={{
              maxWidth: 96,
            }}
          >
            {model}
          </span>
          <SwapOutlined rotate={90} />
        </div>
      </Popover>
    </Dropdown>
  );
  */
};

export default ModelSelector;
