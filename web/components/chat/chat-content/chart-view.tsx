import { AutoChart, BackEndChartType, getChartType } from '@/components/chart/autoChart';
import { formatSql } from '@/utils';
import { Datum } from '@antv/ava';
import { Table, Tabs, TabsProps } from 'antd';
import { CodePreview } from './code-preview';

interface ChartViewProps {
  data: Datum[];
  type: BackEndChartType;
  sql: string;
  id_columns?: string[];
}

function ChartView({ data, type, sql, id_columns }: ChartViewProps) {
  // Helper function to check if a value is numeric
  const isNumeric = (value: any): boolean => {
    if (value === null || value === undefined || value === '') return false;
    if (typeof value === 'number') return true;
    if (typeof value === 'string') {
      const cleaned = value.replace(/[,\s￥$€]/g, '');
      return !isNaN(Number(cleaned)) && cleaned !== '';
    }
    return false;
  };

  // Helper function to check if a column name indicates an ID column
  const isIdColumn = (columnKey: string): boolean => {
    // 优先使用后端传递的 id_columns
    if (id_columns && id_columns.length > 0) {
      return id_columns.includes(columnKey);
    }
    // 备用：使用关键词匹配
    const idKeywords = [
      'id',
      'ID',
      'Id',
      '编号',
      '工号',
      '员工号',
      '学号',
      '订单号',
      '编码',
      '代码',
      '号码',
      '身份证',
      '手机',
      '电话',
      'code',
      'Code',
      'CODE',
      'no',
      'No',
      'NO',
      'num',
      'Num',
      'NUM',
      'number',
      'Number',
    ];
    const keyLower = columnKey.toLowerCase();
    return idKeywords.some(keyword => keyLower.includes(keyword.toLowerCase()));
  };

  // Helper function to determine if a column contains mostly numeric values
  const isNumericColumn = (columnKey: string): boolean => {
    if (!data || data.length === 0) return false;
    const sampleSize = Math.min(10, data.length);
    let numericCount = 0;
    for (let i = 0; i < sampleSize; i++) {
      if (isNumeric(data[i]?.[columnKey])) {
        numericCount++;
      }
    }
    return numericCount > sampleSize * 0.5;
  };

  // 格式化数字为千分位（排除ID列）
  const formatNumber = (value: any): string => {
    if (value === null || value === undefined || value === '') return '';
    const num = Number(value);
    if (isNaN(num)) return String(value);
    return num.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  };

  const columns = data?.[0]
    ? Object.keys(data?.[0])?.map(item => {
        const columnIsNumeric = isNumericColumn(item);
        const columnIsId = isIdColumn(item);
        return {
          title: item,
          dataIndex: item,
          key: item,
          align: columnIsNumeric && !columnIsId ? ('right' as const) : ('left' as const),
          render: (value: any) => {
            // ID列不进行千分位格式化
            if (columnIsId) {
              return value;
            }
            if (columnIsNumeric && isNumeric(value)) {
              return formatNumber(value);
            }
            return value;
          },
        };
      })
    : [];
  const ChartItem = {
    key: 'chart',
    label: 'Chart',
    children: <AutoChart data={data} chartType={getChartType(type)} />,
  };
  const SqlItem = {
    key: 'sql',
    label: 'Code',
    children: <CodePreview language='sql' code={formatSql(sql ?? '', 'mysql') as string} />,
  };
  const DataItem = {
    key: 'data',
    label: 'Data',
    children: <Table dataSource={data} columns={columns} scroll={{ x: 'auto' }} />,
  };
  const TabItems: TabsProps['items'] = type === 'response_table' ? [DataItem, SqlItem] : [DataItem, ChartItem, SqlItem];

  return <Tabs defaultActiveKey={type === 'response_table' ? 'data' : 'data'} items={TabItems} size='small' />;
}

export default ChartView;
