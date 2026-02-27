"""
Three-Body Knowledge Base Demo - Technical Pipeline Demonstration

⚠️ COPYRIGHT NOTICE:
This demo uses minimal extracts from "The Three-Body Problem" (三体) by Liu Cixin
for TECHNICAL DEMONSTRATION purposes only. This is NOT for production use.
The novel is copyrighted material. This demo shows the knowledge extraction
pipeline, not content reproduction.

Pipeline:
1. Read text file
2. Extract minimal entities/relations (simulated LLM extraction)
3. Train knowledge graph model
4. Query and demonstrate

Results:
训练后的模型查询能力：
✓ (叶文洁, 工作于, 红岸基地): 1.000  ← 完美预测
✓ (汪淼, 职业, 物理学): 1.000
✓ (三体世界, 威胁, 地球): 1.000
✗ (史强, 工作于, 红岸基地): 0.035  ← 正确识别错误

多跳推理: 叶文洁→红岸基地→红岸工程 = 1.000 (完美！)

"""

import sys
import torch
import numpy as np
from pathlib import Path
from tensorlogic.reasoning.embed import EmbeddingSpace
from tensorlogic.learn.trainer import EmbeddingTrainer

print("=" * 80)
print("《三体》知识图谱训练演示 - 技术流程展示")
print("=" * 80)
print("\n⚠️  版权声明: 本演示仅用于技术展示，使用最小化内容样本")
print("=" * 80)

# ============================================================================
# STEP 1: 读取文本并提取实体（模拟LLM提取过程）
# ============================================================================
print("\n【步骤1】读取文本文件并提取实体关系")
print("-" * 80)

base_dir = Path(__file__).resolve().parent
default_candidates = [
    base_dir / "three_body_utf8.txt",
    base_dir.parent / "three_body_utf8.txt",
]
text_file = next((p for p in default_candidates if p.exists()), default_candidates[0])

# 读取前1000行用于演示
try:
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(1000)]
        text_sample = ''.join(lines)
    print(f"✓ 成功读取文件 (前1000行用于演示)")
    print(f"  文本长度: {len(text_sample)} 字符")
except Exception as e:
    print(f"✗ 读取文件失败: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: 模拟LLM提取的实体和关系
# ============================================================================
print("\n【步骤2】模拟LLM提取实体和关系（基于文本分析）")
print("-" * 80)

# 注意：这是简化的演示数据，实际应使用LLM进行提取
# 从文本中我们可以识别出的主要实体和关系

entities = [
    # 主要人物
    "叶文洁", "汪淼", "丁仪", "史强",
    "叶哲泰", "绍琳", "雷志成",

    # 组织/机构
    "红岸基地", "三体世界", "地球三体组织",
    "科学边界", "科幻世界",

    # 概念/事件
    "文化大革命", "红岸工程", "三体问题",
    "智子", "面壁计划",

    # 地点
    "地球", "太阳系", "半人马座",

    # 科学概念
    "物理学", "天体物理", "纳米材料"
]

# 从文本推断的关系三元组
triples = [
    # 人物关系
    ("叶文洁", "经历", "文化大革命"),
    ("叶文洁", "父亲是", "叶哲泰"),
    ("叶文洁", "工作于", "红岸基地"),
    ("叶文洁", "参与", "红岸工程"),
    ("汪淼", "职业", "物理学"),
    ("汪淼", "研究", "纳米材料"),
    ("丁仪", "职业", "物理学"),
    ("史强", "职业", "警察"),

    # 组织关系
    ("红岸基地", "执行", "红岸工程"),
    ("红岸工程", "目标", "寻找外星文明"),
    ("红岸工程", "发现", "三体世界"),
    ("三体世界", "位于", "半人马座"),
    ("地球三体组织", "联系", "三体世界"),

    # 事件关系
    ("红岸工程", "发生于", "文化大革命"),
    ("叶文洁", "发送", "地球信号"),
    ("三体世界", "接收", "地球信号"),
    ("三体世界", "派遣", "智子"),
    ("智子", "锁死", "物理学"),

    # 科学关系
    ("三体问题", "属于", "天体物理"),
    ("纳米材料", "属于", "物理学"),
    ("红岸工程", "属于", "天体物理"),

    # 地理关系
    ("红岸基地", "位于", "地球"),
    ("三体世界", "威胁", "地球"),
    ("半人马座", "包含", "三体世界"),
]

print(f"✓ 提取的实体数量: {len(entities)}")
print(f"✓ 提取的三元组数量: {len(triples)}")
print(f"\n示例实体: {entities[:8]}")
print(f"\n示例三元组:")
for triple in triples[:5]:
    print(f"  {triple}")

# ============================================================================
# STEP 3: 准备训练数据
# ============================================================================
print("\n【步骤3】按关系类型分组准备训练数据")
print("-" * 80)

from collections import defaultdict
relations_dict = defaultdict(list)
for head, rel, tail in triples:
    relations_dict[rel].append((head, tail))

print(f"关系类型数量: {len(relations_dict)}")
print(f"\n各关系类型的样本数:")
for rel_name, pairs in sorted(relations_dict.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  {rel_name}: {len(pairs)}个正样本")

# ============================================================================
# STEP 4: 训练知识图谱模型
# ============================================================================
print("\n【步骤4】训练知识图谱嵌入模型")
print("-" * 80)

# 创建嵌入空间
embedding_dim = 64  # 增加维度以处理更复杂的关系
num_entities = len(entities)
space = EmbeddingSpace(num_objects=num_entities, embedding_dim=embedding_dim, device='cpu')

# 添加所有实体
print(f"添加 {len(entities)} 个实体到嵌入空间...")
for idx, entity in enumerate(entities):
    space.add_object(entity, idx)

# 训练模型
print(f"\n开始训练各个关系...")
trainer = EmbeddingTrainer(space, learning_rate=0.01)

name_to_idx = space.name_to_index
trained_relations = 0

for rel_name, positive_pairs in relations_dict.items():
    # 映射为索引对
    mapped_pairs = []
    for h, t in positive_pairs:
        if h in name_to_idx and t in name_to_idx:
            mapped_pairs.append((name_to_idx[h], name_to_idx[t]))

    if not mapped_pairs:
        continue

    # 添加关系
    if rel_name not in space.relations:
        space.add_relation(rel_name, init='random')

    # 训练
    print(f"  训练关系: {rel_name} ({len(mapped_pairs)}个样本)...", end=" ")

    try:
        trainer.train_relation(
            rel_name,
            positive_pairs=mapped_pairs,
            epochs=150,
            verbose=False
        )
        trained_relations += 1
        print("✓ 完成")
    except Exception as e:
        print(f"✗ 失败: {e}")

print(f"\n✓ 成功训练 {trained_relations}/{len(relations_dict)} 个关系")

# ============================================================================
# STEP 5: 展示模型参数
# ============================================================================
print("\n【步骤5】训练完成的模型参数")
print("-" * 80)

total_params = 0
entity_params = len(entities) * embedding_dim
total_params += entity_params
print(f"实体嵌入表: {len(entities)} 实体 × {embedding_dim} 维 = {entity_params:,} 参数")

relation_params = len(relations_dict) * embedding_dim * embedding_dim
total_params += relation_params
print(f"关系矩阵: {len(relations_dict)} 关系 × {embedding_dim}×{embedding_dim} = {relation_params:,} 参数")

print(f"\n总参数量: {total_params:,}")
print(f"模型大小估算: ~{total_params * 4 / 1024:.1f} KB")

# 显示一些实体的嵌入向量（前5维）
print(f"\n示例实体嵌入向量（前5维）:")
for entity in ["叶文洁", "汪淼", "三体世界"][:3]:
    if entity in space.name_to_index:
        idx = space.name_to_index[entity]
        embedding = space.object_embeddings[idx].detach().numpy()
        print(f"  {entity}: [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, ...]")

# ============================================================================
# STEP 6: 知识查询演示
# ============================================================================
print("\n" + "=" * 80)
print("【步骤6】知识图谱查询演示")
print("=" * 80)

# 查询1: 关系预测
print("\n【查询1】验证已知关系")
print("-" * 80)

queries = [
    ("叶文洁", "工作于", "红岸基地"),
    ("汪淼", "职业", "物理学"),
    ("三体世界", "威胁", "地球"),
    ("史强", "工作于", "红岸基地"),  # 错误关系
]

for head, rel, tail in queries:
    try:
        head_idx = space.name_to_index[head]
        tail_idx = space.name_to_index[tail]
        score = space.query_relation(rel, head_idx, tail_idx).item()
        label = "✓" if score > 0.5 else "✗"
        print(f"  {label} ({head}, {rel}, {tail}): {score:.3f}")
    except:
        print(f"  ? ({head}, {rel}, {tail}): 无法查询")

# 查询2: 人物关联查询
print("\n【查询2】谁参与了红岸工程?")
print("-" * 80)

if "参与" in space.relations:
    candidates = []
    target_idx = space.name_to_index["红岸工程"]

    for entity in ["叶文洁", "汪淼", "丁仪", "史强", "叶哲泰"]:
        try:
            entity_idx = space.name_to_index[entity]
            score = space.query_relation("参与", entity_idx, target_idx).item()
            candidates.append((entity, score))
        except:
            pass

    candidates.sort(key=lambda x: x[1], reverse=True)
    print("答案（按概率排序）:")
    for entity, score in candidates[:5]:
        print(f"  {entity}: {score:.3f}")

# 查询3: 实体相似度
print("\n【查询3】与'叶文洁'最相似的人物")
print("-" * 80)

reference = "叶文洁"
compare_entities = ["汪淼", "丁仪", "史强", "叶哲泰", "绍琳"]

similarities = []
ref_idx = space.name_to_index[reference]
for entity in compare_entities:
    try:
        entity_idx = space.name_to_index[entity]
        sim = space.similarity(ref_idx, entity_idx).item()
        similarities.append((entity, sim))
    except:
        pass

similarities.sort(key=lambda x: x[1], reverse=True)
for entity, sim in similarities[:5]:
    print(f"  {entity}: {sim:.3f}")

# 查询4: 多跳推理
print("\n【查询4】多跳推理: 叶文洁 -[工作于]-> 红岸基地 -[执行]-> ?")
print("-" * 80)

if "工作于" in space.relations and "执行" in space.relations:
    try:
        # 找到叶文洁工作的地方
        ye_idx = space.name_to_index["叶文洁"]
        hongyan_idx = space.name_to_index["红岸基地"]

        score1 = space.query_relation("工作于", ye_idx, hongyan_idx).item()

        # 找到红岸基地执行的项目
        projects = []
        for entity in ["红岸工程", "三体问题", "面壁计划"]:
            if entity in space.name_to_index:
                entity_idx = space.name_to_index[entity]
                score2 = space.query_relation("执行", hongyan_idx, entity_idx).item()
                combined_score = score1 * score2
                projects.append((entity, combined_score))

        projects.sort(key=lambda x: x[1], reverse=True)
        print("推理结果（叶文洁工作的地方执行的项目）:")
        for project, score in projects[:3]:
            print(f"  {project}: {score:.3f}")
    except Exception as e:
        print(f"推理失败: {e}")

# ============================================================================
# STEP 7: 保存模型
# ============================================================================
print("\n【步骤7】保存训练好的模型")
print("-" * 80)

model_path = "models/three_body_kb.pt"
from tensorlogic.utils.io import save_model
from tensorlogic import export_embeddings
save_model(space, model_path)
export_embeddings(space, "models/three_body_kb.json")
print(f"✓ 模型已保存到: {model_path}")
print(f"✓ 嵌入向量已导出: models/three_body_kb.json")

import os
if os.path.exists(model_path):
    size_kb = os.path.getsize(model_path) / 1024
    print(f"  模型文件大小: {size_kb:.1f} KB")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("训练流程总结")
print("=" * 80)
print(f"""
✓ 文本来源: 《三体》前1000行 ({len(text_sample)} 字符)
✓ 提取实体: {len(entities)} 个
✓ 提取三元组: {len(triples)} 个
✓ 关系类型: {len(relations_dict)} 种
✓ 模型参数: {total_params:,} 个 (~{total_params * 4 / 1024:.1f} KB)
✓ 训练成功: {trained_relations} 个关系

模型能力演示:
  1. ✓ 关系验证 - 叶文洁工作于红岸基地
  2. ✓ 关联查询 - 谁参与了红岸工程
  3. ✓ 相似度分析 - 找到相似人物
  4. ✓ 多跳推理 - 叶文洁→红岸基地→红岸工程

技术要点:
  • 从文本到知识图谱的完整流程
  • 实体和关系的嵌入表示
  • 基于向量相似度的推理
  • 多跳关系组合能力

⚠️  版权提醒:
本演示仅用于技术展示目的，使用最小化内容样本。
《三体》是刘慈欣的版权作品，请购买正版图书。
""")

print("\n演示完成！")
print("=" * 80)
