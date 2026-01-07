import { ChatContentContext } from '@/pages/chat';
import { useContext } from 'react';
import ExcelDataTable from './excel-data-table';

const ExcelDataTableContainer: React.FC = () => {
  const { excelPreviewData, setExcelPreviewData, setExcelPreviewVisible } = useContext(ChatContentContext);

  const handleDelete = () => {
    setExcelPreviewData?.(undefined);
    setExcelPreviewVisible?.(false);
  };

  return <ExcelDataTable previewData={excelPreviewData} onDelete={handleDelete} />;
};

export default ExcelDataTableContainer;
