import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = {
    "组织": ["回肠"] * 19 + ["盲肠"] * 18,
    "KEGG通路": [
        "D-glutamine and D-glutamate metabolism",
        "Biosynthesis of vancomycin antibiotics",
        "Lysine biosynthesis",
        "Seleno compound metabolism",
        "Biotin metabolism",
        "Lipopolysaccharide biosynthesis",
        "Aminoacyl-tRNA biosynthesis",
        "Carbon fixation pathways in prokaryotes",
        "Peptidoglycan biosynthesis",
        "Thiamine metabolism",
        "Fatty acid biosynthesis",
        "Ribosomes",
        "Homologous recombination",
        "Folic acid-mediated one-carbon metabolism",
        "DNA replication",
        "Streptomycin biosynthesis",
        "Carbon fixation in photosynthetic organisms",
        "Pyrimidine metabolism",
        "Nicotinic acid and nicotinamide metabolism",
        "Ansamycin biosynthesis",
        "D-glutamine and D-glutamate metabolism",
        "Fatty acid biosynthesis",
        "Phenylalanine, tyrosine, and tryptophan biosynthesis",
        "Tryptophan metabolism",
        "Histidine metabolism",
        "Seleno compound metabolism",
        "Peptidoglycan biosynthesis",
        "Pantothenic acid and coenzyme A biosynthesis",
        "Cysteine and methionine metabolism",
        "Folic acid biosynthesis",
        "Streptomycin biosynthesis",
        "Protein export",
        "Lysine biosynthesis",
        "Vancomycin antibiotic biosynthesis",
        "Valine, leucine, and isoleucine biosynthesis",
        "Biotin metabolism",
        "Pentose phosphate pathway"
    ],
    "log2FC": [
        0.23, 0.75, 0.22, 0.17, 0.32, 0.69, 0.25, 0.1, 0.08, 0.1, 0.07, 0.1, 0.11, 0.06, 0.12, 0.41, 0.12, 0.07, -0.22,
        0.42, 0.2, 0.17, 0.17, 0.16, 0.16, 0.08, 0.08, 0.08, 0.08, 0.07, -0.09, -0.16, -0.18, -0.22, -0.4, -0.41, -0.53
    ],
    "p": [
        0, 0.01, 0.03, 0.02, 0.01, 0.04, 0, 0, 0.01, 0, 0, 0, 0.03, 0.02, 0.01, 0, 0, 0.02, 0,
        0, 0.01, 0.05, 0.02, 0, 0.05, 0, 0, 0.01, 0, 0, 0, 0.03, 0.02, 0, 0.01, 0, 0
    ]
}
#
# df = pd.DataFrame(data)
#
# # 处理p值为0的情况（取一个非常小的值代替0，以便于对数转换）
# df['p'] = df['p'].replace(0, 1e-10)
# df['neg_log10_p'] = -np.log10(df['p'])
#
# # 创建图形
# plt.figure(figsize=(12, 12))
#
# # 定义颜色映射
# cmap = plt.cm.viridis_r
# norm = plt.Normalize(0, 5)  # 设置颜色范围
#
# # 为每个组织定义不同的标记
# markers = {'回肠': 'o', '盲肠': 's'}
#
# # 绘制每个点
# for tissue, group in df.groupby('组织'):
#     for _, row in group.iterrows():
#         # 计算点的大小（基于log2FC绝对值）
#         size = 50 + 150 * np.abs(row['log2FC']) / df['log2FC'].abs().max()
#
#         # 绘制散点
#         plt.scatter(
#             row['log2FC'],
#             row['KEGG通路'],
#             s=size,
#             c=row['neg_log10_p'],
#             cmap=cmap,
#             norm=norm,
#             marker=markers[tissue],
#             edgecolor='k',
#             linewidth=0.5,
#             label=tissue if _ == 0 else ""  # 避免重复标签
#         )
#
#         # 添加p值标注
#         p_text = f"{row['p']:.1e}" if row['p'] >= 1e-3 else "<1e-3"
#         ha = 'left' if row['log2FC'] >= 0 else 'right'
#         offset = 0.02 if row['log2FC'] >= 0 else -0.02
#         plt.text(
#             row['log2FC'] + offset,
#             row['KEGG通路'],
#             p_text,
#             fontsize=8,
#             ha=ha,
#             va='center',
#             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
#         )
#
# # 添加垂直线
# plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
#
# # 添加颜色条
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, pad=0.02)
# cbar.set_label('-log10(p-value)')
#
# # 添加图例
# handles, labels = plt.gca().get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# plt.legend(unique_labels.values(), unique_labels.keys(),
#            title='组织', bbox_to_anchor=(1.05, 1), loc='upper left')
#
# # 设置标题和标签
# plt.title("KEGG Pathway Enrichment Analysis", pad=20)
# plt.xlabel("log2 Fold Change")
# plt.ylabel("KEGG Pathway")
#
# # 调整布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()

df = pd.DataFrame(data)

# 处理p值为0的情况
df['p'] = df['p'].replace(0, 1e-10)
df['neg_log10_p'] = -np.log10(df['p'])

# 提取回肠数据
# ileum_df = df[df['组织'] == '回肠']
ileum_df = df[df['组织'] == '盲肠']
# 创建图形
plt.figure(figsize=(10, 12))

# 定义颜色映射
cmap = plt.cm.viridis_r
norm = plt.Normalize(0, 5)

# 绘制回肠数据
for _, row in ileum_df.iterrows():
    size = 50 + 150 * np.abs(row['log2FC']) / df['log2FC'].abs().max()
    plt.scatter(
        row['log2FC'],
        row['KEGG通路'],
        s=size,
        c=row['neg_log10_p'],
        cmap=cmap,
        norm=norm,
        marker='o',
        edgecolor='k',
        linewidth=0.5
    )
    # # 添加p值标注（排除靠近原点的点）
    # if abs(row['log2FC']) > 0.1:  # 只给log2FC绝对值大于0.1的点加标注
    #     p_text = f"{row['p']:.1e}" if row['p'] >= 1e-3 else "<1e-3"
    #     ha = 'left' if row['log2FC'] >= 0 else 'right'
    #     offset = 0.02 if row['log2FC'] >= 0 else -0.02
    #     plt.text(
    #         row['log2FC'] + offset,
    #         row['KEGG通路'],
    #         p_text,
    #         fontsize=8,
    #         ha=ha,
    #         va='center',
    #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    #     )

# 添加参考线
# plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02)
cbar.set_label('-log10(p-value)')

# 设置标题和标签
plt.title("回肠 KEGG Pathway Enrichment Analysis")
plt.xlabel("log2 Fold Change")
plt.ylabel("KEGG Pathway")

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()