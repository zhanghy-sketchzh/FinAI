import { BackEndChartType } from '@/components/chart';
import { Datum } from '@antv/ava';
import ChartView from './chart-view';

interface Props {
  data: {
    data: Datum[];
    describe: string;
    title: string;
    type: BackEndChartType;
    sql: string;
    id_columns?: string[];
  };
}

function VisChart({ data }: Props) {
  if (!data) {
    return null;
  }
  return <ChartView data={data?.data} type={data?.type} sql={data?.sql} id_columns={data?.id_columns} />;
}

export default VisChart;
