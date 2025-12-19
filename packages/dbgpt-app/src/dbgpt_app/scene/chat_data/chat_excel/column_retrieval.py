"""
列检索模块 - 基于混合检索（BM25 + 向量）召回相关列信息
"""
import json
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import heapq
from typing import List, Dict, Any


class ColumnRetrieval:
    """列检索类 - 从schema_json中提取列信息并进行混合检索"""
    
    def __init__(self, schema_json_str: str):
        """
        初始化列检索
        
        Args:
            schema_json_str: JSON格式的schema字符串
        """
        self.schema_data = json.loads(schema_json_str)
        self.columns = self.schema_data.get('columns', [])
        
        # 构建检索所需的数据
        self.column_names = []
        self.column_descriptions = []  # 用于检索的文本
        self.column_info_map = {}  # 列名 -> 完整信息的映射
        
        self._prepare_data()
        
        # 构建BM25索引
        self.tokenized_docs = [jieba.lcut(desc) for desc in self.column_descriptions]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _prepare_data(self):
        """准备检索数据"""
        for col in self.columns:
            col_name = col.get('column_name', '')
            self.column_names.append(col_name)
            
            # 构建用于检索的描述文本（组合多个字段）
            desc_parts = [
                col_name,
                col.get('description', ''),
                col.get('semantic_type', ''),
                ' '.join(col.get('analysis_usage', [])),
            ]
            
            # 如果有unique_values，也加入
            if 'unique_values' in col:
                desc_parts.append(' '.join(str(v) for v in col['unique_values'][:10]))
            
            description = ' '.join(filter(None, desc_parts))
            self.column_descriptions.append(description)
            
            # 保存完整信息
            self.column_info_map[col_name] = col
    
    def _bm25_retrieve(self, query: str, k: int = 10) -> Dict[str, float]:
        """BM25检索"""
        query_tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top k
        top_indices = np.argsort(scores)[-k:][::-1]
        return {self.column_names[i]: float(scores[i]) for i in top_indices}
    
    def _keyword_match_retrieve(self, query: str, k: int = 10) -> Dict[str, float]:
        """
        关键词匹配检索（作为向量检索的简化替代）
        计算query中的关键词在列描述中出现的次数
        """
        query_tokens = set(jieba.lcut(query.lower()))
        scores = {}
        
        for col_name, desc in zip(self.column_names, self.column_descriptions):
            desc_tokens = set(jieba.lcut(desc.lower()))
            # 计算交集
            intersection = query_tokens & desc_tokens
            score = len(intersection) / max(len(query_tokens), 1)
            scores[col_name] = score
        
        # 返回top k
        top_items = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
        return dict(top_items)
    
    @staticmethod
    def _rrf_fusion(bm25_scores: Dict[str, float], 
                    keyword_scores: Dict[str, float], 
                    k: int = 60) -> Dict[str, float]:
        """
        RRF融合算法
        
        Args:
            bm25_scores: BM25得分
            keyword_scores: 关键词匹配得分
            k: RRF参数
        
        Returns:
            融合后的得分
        """
        all_keys = set(bm25_scores.keys()).union(keyword_scores.keys())
        fused_scores = {}
        
        for key in all_keys:
            bm25_score = bm25_scores.get(key, 0)
            keyword_score = keyword_scores.get(key, 0)
            
            # RRF公式
            fused_scores[key] = 1 / (k + bm25_score) + 1 / (k + keyword_score)
        
        return fused_scores
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索 - 召回最相关的top_k个列
        
        Args:
            query: 查询文本（通常是改写后的query）
            top_k: 返回的列数量
        
        Returns:
            相关列的详细信息列表
        """
        # BM25检索
        bm25_scores = self._bm25_retrieve(query, k=top_k * 2)
        
        # 关键词匹配检索
        keyword_scores = self._keyword_match_retrieve(query, k=top_k * 2)
        
        # RRF融合
        fused_scores = self._rrf_fusion(bm25_scores, keyword_scores)
        
        # 获取top k
        top_columns = heapq.nlargest(top_k, fused_scores.items(), key=lambda x: x[1])
        
        # 返回详细信息
        results = []
        for col_name, score in top_columns:
            col_info = self.column_info_map.get(col_name, {}).copy()
            col_info['retrieval_score'] = score
            results.append(col_info)
        
        return results
    
    def format_retrieved_columns(self, retrieved_columns: List[Dict[str, Any]]) -> str:
        """
        格式化检索到的列信息为可读文本
        
        Args:
            retrieved_columns: 检索到的列信息列表
        
        Returns:
            格式化后的文本
        """
        if not retrieved_columns:
            return ""
        
        lines = ["=== 最相关的列信息 ==="]
        
        for i, col in enumerate(retrieved_columns, 1):
            lines.append(f"\n【列{i}】{col.get('column_name', '')}")
            lines.append(f"  数据类型: {col.get('data_type', '')}")
            lines.append(f"  语义类型: {col.get('semantic_type', '')}")
            lines.append(f"  描述: {col.get('description', '')}")
            
            # 分析用法
            if 'analysis_usage' in col:
                lines.append(f"  分析用法: {', '.join(col['analysis_usage'])}")
            
            # 唯一值（分类字段）
            if 'unique_values' in col and col['unique_values']:
                unique_vals = col['unique_values'][:10]  # 最多显示10个
                lines.append(f"  唯一值示例: {', '.join(str(v) for v in unique_vals)}")
                if 'value_distribution' in col:
                    lines.append(f"  分布说明: {col['value_distribution']}")
            
            # 统计摘要（数值字段）
            if 'statistics_summary' in col:
                lines.append(f"  统计信息: {col['statistics_summary']}")
            
            # 是否关键字段
            if col.get('is_key_field'):
                lines.append(f"  ⭐ 关键字段")
        
        return '\n'.join(lines)


def test_column_retrieval():
    """测试函数"""
    # 模拟schema_json
    schema_json = {
        "table_name": "sales_data",
        "table_description": "销售数据表",
        "columns": [
            {
                "column_name": "订单日期",
                "data_type": "DATE",
                "semantic_type": "时间维度",
                "description": "订单生成的日期，用于时间序列分析",
                "is_key_field": False,
                "analysis_usage": ["分组", "筛选", "排序"],
                "value_distribution": "时间跨度覆盖2020-2023年"
            },
            {
                "column_name": "区域",
                "data_type": "VARCHAR",
                "semantic_type": "地域维度",
                "description": "客户所在区域（华东、中南、东北等）",
                "is_key_field": False,
                "analysis_usage": ["分组", "筛选"],
                "unique_values": ["华东", "中南", "东北", "华北", "西南", "西北"],
                "value_distribution": "华东和中南区域订单量最高"
            },
            {
                "column_name": "利润",
                "data_type": "DOUBLE",
                "semantic_type": "数值指标",
                "description": "订单的利润金额",
                "is_key_field": False,
                "analysis_usage": ["聚合", "排序"],
                "statistics_summary": "最小值-500，最大值10000，平均值200"
            },
            {
                "column_name": "产品名称",
                "data_type": "VARCHAR",
                "semantic_type": "分类维度",
                "description": "产品的名称",
                "is_key_field": False,
                "analysis_usage": ["分组", "筛选"]
            }
        ]
    }
    
    # 创建检索器
    retriever = ColumnRetrieval(json.dumps(schema_json))
    
    # 测试检索
    query = "2022年8月华北地区的利润同比环比分析"
    results = retriever.retrieve(query, top_k=3)
    
    print("检索结果:")
    for col in results:
        print(f"- {col['column_name']} (得分: {col['retrieval_score']:.4f})")
    
    print("\n格式化输出:")
    print(retriever.format_retrieved_columns(results))


if __name__ == "__main__":
    test_column_retrieval()



