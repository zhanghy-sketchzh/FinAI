import { AutoChart, BackEndChartType, getChartType } from '@/components/chart/autoChart';
import { formatSql } from '@/utils';
import { Datum } from '@antv/ava';
import { Table, Tabs, TabsProps } from 'antd';
import { CodePreview } from './code-preview';

function ChartView({ data, type, sql }: { data: Datum[]; type: BackEndChartType; sql: string }) {
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

  // 格式化数字为千分位
  const formatNumber = (value: any): string => {
    if (value === null || value === undefined || value === '') return '';
    const num = Number(value);
    if (isNaN(num)) return String(value);
    return num.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  };

  const columns = data?.[0]
    ? Object.keys(data?.[0])?.map(item => {
        const columnIsNumeric = isNumericColumn(item);
        return {
          title: item,
          dataIndex: item,
          key: item,
          align: columnIsNumeric ? 'right' : 'left',
          render: (value: any) => {
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
