import { CloseOutlined, FilterOutlined, SearchOutlined, TableOutlined } from '@ant-design/icons';
import { Button, Input, Popover, Spin, Table, Tag, Tooltip } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { ColumnType, FilterValue, SorterResult } from 'antd/es/table/interface';
import { useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface ExcelDataTableProps {
  previewData?: {
    columns: Array<{ field: string; type: string; headerName: string }>;
    rows: Array<Record<string, any>>;
    total: number;
    file_name?: string;
  };
  onDelete?: () => void;
}

const ExcelDataTable: React.FC<ExcelDataTableProps> = ({ previewData, onDelete }) => {
  const { t } = useTranslation();
  const [searchText, setSearchText] = useState<Record<string, string>>({});
  const [sortedInfo, setSortedInfo] = useState<SorterResult<Record<string, any>>>({});
  const [filterVisible, setFilterVisible] = useState<Record<string, boolean>>({});
  const searchInputRef = useRef<any>(null);

  // 格式化数字为千分位
  const formatNumber = (value: any): string => {
    if (value === null || value === undefined || value === '') return '-';
    const num = Number(value);
    if (isNaN(num)) return String(value);
    return num.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  };

  const handleTableChange = (
    _pagination: any,
    _filters: Record<string, FilterValue | null>,
    sorter: SorterResult<Record<string, any>> | SorterResult<Record<string, any>>[],
  ) => {
    setSortedInfo(Array.isArray(sorter) ? sorter[0] : sorter);
  };

  // 过滤后的数据
  const filteredData = useMemo(() => {
    if (!previewData?.rows) return [];
    return previewData.rows.filter(row => {
      return Object.entries(searchText).every(([key, value]) => {
        if (!value) return true;
        const cellValue = row[key];
        return String(cellValue ?? '')
          .toLowerCase()
          .includes(value.toLowerCase());
      });
    });
  }, [previewData?.rows, searchText]);

  // 活跃筛选数量
  const activeFilterCount = useMemo(() => {
    return Object.values(searchText).filter(v => v && v.trim()).length;
  }, [searchText]);

  if (!previewData || !previewData.columns || previewData.columns.length === 0) {
    return (
      <div className='flex flex-col items-center justify-center h-full gap-4 bg-[#ffffff80] dark:bg-[#ffffff29]'>
        <div className='relative'>
          <div className='absolute inset-0 bg-blue-500/20 blur-xl rounded-full' />
          <Spin size='large' />
        </div>
        <span className='text-slate-500 dark:text-slate-400 text-sm tracking-wide'>{t('loading_data')}</span>
      </div>
    );
  }

  // 根据列名长度和数据内容计算合适的列宽
  const getColumnWidth = (col: { field: string; headerName: string }) => {
    const headerLen = col.headerName.length;
    const maxDataLen = Math.max(
      ...filteredData.slice(0, 100).map(row => String(row[col.field] ?? '').length),
      headerLen,
    );
    const estimatedWidth = Math.max(headerLen * 14, maxDataLen * 10);
    const width = Math.max(120, Math.min(estimatedWidth + 60, 350));
    return width;
  };

  // 获取类型标签颜色
  const getTypeColor = (type: string) => {
    const t = type.toLowerCase();
    if (t.includes('int') || t.includes('float') || t.includes('double') || t.includes('number')) {
      return { bg: 'bg-emerald-50 dark:bg-emerald-900/30', text: 'text-emerald-600 dark:text-emerald-400' };
    }
    if (t.includes('date') || t.includes('time')) {
      return { bg: 'bg-amber-50 dark:bg-amber-900/30', text: 'text-amber-600 dark:text-amber-400' };
    }
    return { bg: 'bg-blue-50 dark:bg-blue-900/30', text: 'text-blue-600 dark:text-blue-400' };
  };

  // 转换列定义
  const columns: ColumnsType<Record<string, any>> = previewData.columns.map(col => {
    const isNumeric =
      col.type.toLowerCase().includes('int') ||
      col.type.toLowerCase().includes('float') ||
      col.type.toLowerCase().includes('double');

    const hasFilter = !!searchText[col.field];
    const colWidth = getColumnWidth(col);
    const typeStyle = getTypeColor(col.type);

    const column: ColumnType<Record<string, any>> = {
      title: (
        <div className='flex flex-col gap-0.5 py-1'>
          <div className='flex items-center justify-between gap-2'>
            <Tooltip title={col.headerName} placement='topLeft'>
              <span className='font-semibold text-slate-700 dark:text-slate-200 truncate text-[13px] tracking-tight'>
                {col.headerName}
              </span>
            </Tooltip>
            <Popover
              open={filterVisible[col.field]}
              onOpenChange={visible => {
                setFilterVisible(prev => ({ ...prev, [col.field]: visible }));
                if (visible) {
                  setTimeout(() => searchInputRef.current?.focus(), 100);
                }
              }}
              trigger='click'
              placement='bottomRight'
              content={
                <div className='flex flex-col gap-3 p-1' style={{ width: 180 }}>
                  <Input
                    ref={searchInputRef}
                    size='middle'
                    placeholder={t('filter_placeholder')}
                    value={searchText[col.field] || ''}
                    onChange={e => setSearchText(prev => ({ ...prev, [col.field]: e.target.value }))}
                    onPressEnter={() => setFilterVisible(prev => ({ ...prev, [col.field]: false }))}
                    allowClear
                    prefix={<SearchOutlined className='text-slate-400' />}
                    className='rounded-lg'
                  />
                  <div className='flex justify-end gap-2'>
                    <Button
                      size='small'
                      onClick={() => {
                        setSearchText(prev => ({ ...prev, [col.field]: '' }));
                        setFilterVisible(prev => ({ ...prev, [col.field]: false }));
                      }}
                      className='rounded-md'
                    >
                      {t('reset')}
                    </Button>
                    <Button
                      type='primary'
                      size='small'
                      onClick={() => setFilterVisible(prev => ({ ...prev, [col.field]: false }))}
                      className='rounded-md'
                    >
                      {t('confirm')}
                    </Button>
                  </div>
                </div>
              }
            >
              <div
                className={`
                  w-5 h-5 rounded flex items-center justify-center cursor-pointer transition-all duration-200
                  ${
                    hasFilter
                      ? 'bg-blue-500 text-white shadow-sm'
                      : 'hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-400 hover:text-blue-500'
                  }
                `}
                onClick={e => e.stopPropagation()}
              >
                <FilterOutlined className='text-[10px]' />
              </div>
            </Popover>
          </div>
          <span className={`text-[10px] px-1.5 py-0.5 rounded w-fit ${typeStyle.bg} ${typeStyle.text}`}>
            {col.type}
          </span>
        </div>
      ),
      dataIndex: col.field,
      key: col.field,
      width: colWidth,
      ellipsis: { showTitle: true },
      align: isNumeric ? 'right' : 'left',
      render: (value: any) => {
        if (value === null || value === undefined || value === '') {
          return <span className='text-slate-300 dark:text-slate-600'>-</span>;
        }
        if (isNumeric) {
          return (
            <span className='font-mono text-[13px] text-slate-700 dark:text-slate-300 tabular-nums'>
              {formatNumber(value)}
            </span>
          );
        }
        return <span className='text-[13px] text-slate-600 dark:text-slate-400'>{value}</span>;
      },
      sorter: (a, b) => {
        const aVal = a[col.field];
        const bVal = b[col.field];
        if (isNumeric) {
          return (Number(aVal) || 0) - (Number(bVal) || 0);
        }
        return String(aVal ?? '').localeCompare(String(bVal ?? ''));
      },
      sortOrder: sortedInfo.columnKey === col.field ? sortedInfo.order : null,
      onHeaderCell: () => ({
        style: {
          minWidth: colWidth,
          maxWidth: colWidth,
        },
      }),
      onCell: () => ({
        style: {
          minWidth: colWidth,
          maxWidth: colWidth,
        },
      }),
    };
    return column;
  });

  const fileName = previewData?.file_name || '';
  const rowCount = previewData?.total || 0;
  const columnCount = previewData?.columns?.length || 0;

  return (
    <div className='h-full flex flex-col bg-[#ffffff80] dark:bg-[#ffffff29]'>
      {/* 头部区域 */}
      <div className='flex-none px-6 py-4 border-b border-[#d5e5f6] dark:border-[#ffffff66] bg-[#ffffff99] dark:bg-[rgba(255,255,255,0.1)] backdrop-blur-sm'>
        <div className='flex items-center justify-between'>
          <div className='flex items-center gap-4'>
            <div className='flex items-center gap-3'>
              <div className='w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/25'>
                <TableOutlined className='text-white text-base' />
              </div>
              <div className='flex flex-col'>
                <span className='text-base font-semibold text-slate-800 dark:text-slate-100 tracking-tight'>
                  {t('data_preview')}
                </span>
                {fileName && (
                  <Tooltip title={fileName}>
                    <span className='text-xs text-slate-500 dark:text-slate-400 truncate max-w-[200px]'>
                      {fileName}
                    </span>
                  </Tooltip>
                )}
              </div>
            </div>
          </div>

          <div className='flex items-center gap-3'>
            {activeFilterCount > 0 && (
              <Tag
                color='blue'
                className='rounded-full px-3 py-0.5 text-xs font-medium border-0 bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
              >
                {activeFilterCount} 个筛选
              </Tag>
            )}
            <div className='flex items-center gap-2 px-4 py-2 rounded-xl bg-white/60 dark:bg-[rgba(255,255,255,0.1)]'>
              <div className='flex items-center gap-1.5'>
                <span className='text-lg font-bold text-blue-600 dark:text-blue-400 tabular-nums'>
                  {rowCount.toLocaleString()}
                </span>
                <span className='text-xs text-slate-500 dark:text-slate-400'>{t('rows')}</span>
              </div>
              <span className='text-slate-300 dark:text-slate-600'>×</span>
              <div className='flex items-center gap-1.5'>
                <span className='text-lg font-bold text-emerald-600 dark:text-emerald-400 tabular-nums'>
                  {columnCount}
                </span>
                <span className='text-xs text-slate-500 dark:text-slate-400'>{t('columns')}</span>
              </div>
            </div>
            {onDelete && (
              <Tooltip title='删除数据预览'>
                <Button
                  type='text'
                  icon={<CloseOutlined />}
                  onClick={onDelete}
                  className='flex items-center justify-center w-8 h-8 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 text-slate-500 hover:text-red-500 dark:text-slate-400 dark:hover:text-red-400 transition-colors'
                />
              </Tooltip>
            )}
          </div>
        </div>
      </div>

      {/* 表格区域 */}
      <div className='flex-1 min-h-0 overflow-hidden'>
        <Table
          columns={columns}
          dataSource={filteredData}
          rowKey={(_, index) => String(index)}
          onChange={handleTableChange}
          pagination={{
            defaultPageSize: 100,
            showSizeChanger: true,
            pageSizeOptions: ['50', '100', '200', '500'],
            showQuickJumper: true,
            showTotal: total => (
              <span className='text-slate-500 dark:text-slate-400 text-sm'>共 {total.toLocaleString()} 条记录</span>
            ),
            className: '!mb-0 !pb-0 !px-4',
          }}
          scroll={{ x: true, y: 'calc(100vh - 200px)' }}
          size='middle'
          className='
            h-full excel-preview-table
            [&_.ant-table-wrapper]:h-full 
            [&_.ant-table]:h-full 
            [&_.ant-table-container]:h-full
            [&_.ant-table-container]:!bg-transparent 
            [&_.ant-table-thead>tr>th]:!bg-white/80
            [&_.ant-table-thead>tr>th]:dark:!bg-[rgba(255,255,255,0.12)]
            [&_.ant-table-thead>tr>th]:!border-b-2
            [&_.ant-table-thead>tr>th]:!border-[#d5e5f6]
            [&_.ant-table-thead>tr>th]:dark:!border-[#ffffff66]
            [&_.ant-table-tbody>tr>td]:!bg-white/60
            [&_.ant-table-tbody>tr>td]:dark:!bg-[rgba(255,255,255,0.08)]
            [&_.ant-table-tbody>tr:hover>td]:!bg-blue-50/50
            [&_.ant-table-tbody>tr:hover>td]:dark:!bg-[rgba(255,255,255,0.12)]
            [&_.ant-table-tbody>tr>td]:!border-slate-100
            [&_.ant-table-tbody>tr>td]:dark:!border-slate-800
            [&_.ant-table-tbody>tr>td]:!py-2.5
            [&_.ant-pagination]:!mt-0 
            [&_.ant-pagination]:!py-3
            [&_.ant-pagination]:!bg-[#ffffff99]
            [&_.ant-pagination]:dark:!bg-[rgba(255,255,255,0.1)]
            [&_.ant-pagination]:!backdrop-blur-sm
            [&_.ant-pagination]:!border-t 
            [&_.ant-pagination]:!border-[#d5e5f6]
            [&_.ant-pagination]:dark:!border-[#ffffff66]
            [&_.ant-table-column-sorter]:!text-slate-400
            [&_.ant-table-column-sorter-up.active]:!text-blue-500
            [&_.ant-table-column-sorter-down.active]:!text-blue-500
          '
        />
      </div>
    </div>
  );
};

export default ExcelDataTable;
