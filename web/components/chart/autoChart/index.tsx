import { ChatContext } from '@/app/chat-context';
import i18n, { I18nKeys } from '@/app/i18n';
import { DownloadOutlined } from '@ant-design/icons';
import { Advice, Advisor, Datum } from '@antv/ava';
import { Chart, ChartRef } from '@berryv/g2-react';
import { Button, Col, Empty, Row, Select, Space, Tooltip } from 'antd';
import { compact, concat, uniq } from 'lodash';
import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { downloadImage } from '../helpers/downloadChartImage';
import { customizeAdvisor, getVisAdvices } from './advisor/pipeline';
import { defaultAdvicesFilter } from './advisor/utils';
import { customCharts } from './charts';
import { processNilData, sortData } from './charts/util';
import { AutoChartProps, ChartType, CustomAdvisorConfig, CustomChart, Specification } from './types';
const { Option } = Select;

export const AutoChart = (props: AutoChartProps) => {
  const { data: originalData, chartType, scopeOfCharts, ruleConfig } = props;
  // 处理空值数据 (为'-'的数据)
  const data = processNilData(originalData) as Datum[];
  const { mode } = useContext(ChatContext);

  const [advisor, setAdvisor] = useState<Advisor>();
  const [advices, setAdvices] = useState<Advice[]>([]);
  const [renderChartType, setRenderChartType] = useState<ChartType>();
  const chartRef = useRef<ChartRef>();
  const prevAdvicesRef = useRef<string>(''); // 用于跟踪advices的变化

  useEffect(() => {
    const input_charts: CustomChart[] = customCharts;
    const advisorConfig: CustomAdvisorConfig = {
      charts: input_charts,
      scopeOfCharts: {
        // 排除面积图
        exclude: ['area_chart', 'stacked_area_chart', 'percent_stacked_area_chart'],
      },
      ruleConfig,
    };
    setAdvisor(customizeAdvisor(advisorConfig));
  }, [ruleConfig, scopeOfCharts]);

  /** 将 AVA 得到的图表推荐结果和模型的合并 */
  const getMergedAdvices = (avaAdvices: Advice[]) => {
    if (!advisor) return [];
    const filteredAdvices = defaultAdvicesFilter({
      advices: avaAdvices,
    });
    const allChartTypes = uniq(
      compact(
        concat(
          chartType,
          avaAdvices.map(item => item.type),
        ),
      ),
    );
    const allAdvices = allChartTypes
      .map(chartTypeItem => {
        const avaAdvice = filteredAdvices.find(item => item.type === chartTypeItem);
        // 如果在 AVA 推荐列表中，直接采用推荐列表中的结果
        if (avaAdvice) {
          return avaAdvice;
        }
        // 如果不在，则单独为其生成图表 spec
        const dataAnalyzerOutput = advisor.dataAnalyzer.execute({ data });
        if ('data' in dataAnalyzerOutput) {
          const specGeneratorOutput = advisor.specGenerator.execute({
            data: dataAnalyzerOutput.data,
            dataProps: dataAnalyzerOutput.dataProps,
            chartTypeRecommendations: [{ chartType: chartTypeItem, score: 1 }],
          });
          if ('advices' in specGeneratorOutput) return specGeneratorOutput.advices?.[0];
        }
      })
      .filter(advice => advice?.spec) as Advice[];
    return allAdvices;
  };

  useEffect(() => {
    if (data && advisor) {
      const avaAdvices = getVisAdvices({
        data,
        myChartAdvisor: advisor,
      });
      // 合并模型推荐的图表类型和 ava 推荐的图表类型
      const allAdvices = getMergedAdvices(avaAdvices);

      // 生成advices的唯一标识，用于判断advices是否真正变化
      const advicesKey = JSON.stringify(allAdvices.map(a => a.type));
      const advicesChanged = advicesKey !== prevAdvicesRef.current;

      setAdvices(allAdvices);

      // 只在以下情况设置默认图表类型：
      // 1. renderChartType 未设置（首次加载）
      // 2. advices 真正发生变化且当前选择的图表类型不在新列表中
      setRenderChartType(currentType => {
        if (!currentType) {
          // 首次加载，设置默认值
          prevAdvicesRef.current = advicesKey;
          return allAdvices[0]?.type as ChartType;
        } else if (advicesChanged) {
          // advices变化了，检查当前选择的类型是否仍然有效
          const currentTypeStillValid = allAdvices.some(a => a.type === currentType);
          if (!currentTypeStillValid) {
            // 当前选择的类型无效，重置为第一个
            prevAdvicesRef.current = advicesKey;
            return allAdvices[0]?.type as ChartType;
          }
          // 当前类型仍然有效，保持不变
          prevAdvicesRef.current = advicesKey;
          return currentType;
        }
        // advices没有变化，保持当前选择
        return currentType;
      });
    }
  }, [JSON.stringify(data), advisor, chartType]);

  const visComponent = useMemo(() => {
    /* Advices exist, render the chart. */
    if (advices?.length > 0) {
      const chartTypeInput = renderChartType ?? advices[0].type;
      const spec: Specification = advices?.find((item: Advice) => item.type === chartTypeInput)?.spec ?? undefined;
      if (spec) {
        if (spec.data && ['line_chart', 'step_line_chart'].includes(chartTypeInput)) {
          // 处理 ava 内置折线图的排序问题
          const dataAnalyzerOutput = advisor?.dataAnalyzer.execute({ data });
          if (dataAnalyzerOutput && 'dataProps' in dataAnalyzerOutput) {
            spec.data = sortData({
              data: spec.data,
              xField: dataAnalyzerOutput.dataProps?.find((field: any) => field.recommendation === 'date'),
              chartType: chartTypeInput,
            });
          }
        }
        if (chartTypeInput === 'pie_chart' && spec?.encode?.color) {
          // 补充饼图的 tooltip title 展示
          spec.tooltip = { title: { field: spec.encode.color } };
        }
        return (
          <Chart
            key={chartTypeInput}
            options={{
              ...spec,
              autoFit: true,
              theme: mode,
              height: 450,
            }}
            ref={chartRef}
          />
        );
      }
    }
  }, [advices, mode, renderChartType]);

  if (renderChartType) {
    return (
      <div>
        <Row justify='space-between' className='mb-2'>
          <Col>
            <Space>
              <span>{i18n.t('Advices')}</span>
              <Select
                className='w-52'
                value={renderChartType}
                placeholder={'Chart Switcher'}
                onChange={value => setRenderChartType(value)}
                size={'small'}
              >
                {advices?.map(item => {
                  const name = i18n.t(item.type as I18nKeys);
                  return (
                    <Option key={item.type} value={item.type}>
                      <Tooltip title={name} placement={'right'}>
                        <div>{name}</div>
                      </Tooltip>
                    </Option>
                  );
                })}
              </Select>
            </Space>
          </Col>
          <Col>
            <Tooltip title={i18n.t('Download')}>
              <Button
                onClick={() => downloadImage(chartRef.current, i18n.t(renderChartType as I18nKeys))}
                icon={<DownloadOutlined />}
                type='text'
              />
            </Tooltip>
          </Col>
        </Row>
        <div className='flex pb-4'>{visComponent}</div>
      </div>
    );
  }

  return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={i18n.t('no_suitable_visualization')} />;
};

export * from './helpers';
