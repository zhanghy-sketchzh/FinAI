import { ChatContentContext } from '@/pages/chat';
import { useContext } from 'react';
import ExcelDataTable from './excel-data-table';

const ExcelDataTableContainer: React.FC = () => {
  const { excelPreviewData, setExcelPreviewVisible } = useContext(ChatContentContext);

  const handleDelete = () => {
    // 只隐藏预览面板，不清除数据，这样用户可以再次点击眼睛图标重新显示
    setExcelPreviewVisible?.(false);
  };

  return <ExcelDataTable previewData={excelPreviewData} onDelete={handleDelete} />;
};

export default ExcelDataTableContainer;
