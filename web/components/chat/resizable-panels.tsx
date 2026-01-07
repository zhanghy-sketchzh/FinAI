import { useCallback, useEffect, useRef, useState } from 'react';

interface ResizablePanelsProps {
  leftPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  defaultLeftWidth?: number; // 百分比，默认60
  minLeftWidth?: number; // 最小宽度百分比，默认30
  maxLeftWidth?: number; // 最大宽度百分比，默认80
}

const ResizablePanels: React.FC<ResizablePanelsProps> = ({
  leftPanel,
  rightPanel,
  defaultLeftWidth = 60,
  minLeftWidth = 30,
  maxLeftWidth = 80,
}) => {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const container = containerRef.current;
      const containerRect = container.getBoundingClientRect();
      const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

      // 限制在最小和最大宽度之间
      if (newLeftWidth >= minLeftWidth && newLeftWidth <= maxLeftWidth) {
        setLeftWidth(newLeftWidth);
      }
    },
    [isDragging, minLeftWidth, maxLeftWidth],
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      // 禁止文本选择
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div ref={containerRef} className='flex h-full w-full'>
      {/* 左侧面板 */}
      <div
        className='h-full overflow-hidden'
        style={{
          width: `${leftWidth}%`,
          transition: isDragging ? 'none' : 'width 0.1s ease',
        }}
      >
        {leftPanel}
      </div>

      {/* 可拖动的分隔条 */}
      <div
        className='relative flex-shrink-0 group cursor-col-resize'
        onMouseDown={handleMouseDown}
        style={{
          width: '4px',
        }}
      >
        {/* 分隔线 */}
        <div className='absolute inset-0 bg-gray-200 dark:bg-gray-700 group-hover:bg-blue-400 dark:group-hover:bg-blue-500 transition-colors' />

        {/* 可拖动的把手（hover时显示） */}
        <div className='absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-1 h-12 bg-gray-400 dark:bg-gray-500 group-hover:bg-blue-500 dark:group-hover:bg-blue-400 rounded-full opacity-0 group-hover:opacity-100 transition-opacity' />
      </div>

      {/* 右侧面板 */}
      <div
        className='h-full overflow-hidden'
        style={{
          width: `${100 - leftWidth}%`,
          transition: isDragging ? 'none' : 'width 0.1s ease',
        }}
      >
        {rightPanel}
      </div>
    </div>
  );
};

export default ResizablePanels;
