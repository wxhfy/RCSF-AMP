import torch

# 替换为你的pt/pth文件路径
# pt_path = "/home/fyh0106/work/full_scale_ablation_results/checkpoints/baseline/best_2.0.pth"
# pt_path = "/home/20T-1/fyh0106/compare2/merged_amp_decoy/checkpoints/no_mamba/best_2.0.pth"
# pt_path = "/home/20T-1/fyh0106/compare2/merged_amp_decoy/checkpoints/no_plddt/best_2.0.pth"
# pt_path = "/home/20T-1/fyh0106/compare2/merged_amp_decoy/graphs/test_042L_FRG3G.pt"
pt_path = "/home/fyh0106/SUCF/benchmark1_graph/graphs/AP00001.pt"
# pt_path = "/home/fyh0106/SUCF/benchmark2_graph/graphs/2B_TAV.pt"
# pt_path = "/home/20T-1/fyh0106/compare/merged_amp_decoy/graphs/test_AP00001.pt"
ckpt = torch.load(pt_path, map_location="cpu")

print("== 顶层key ==")
for k in ckpt.keys():
    print(f"  {k}")
print(ckpt["x"])
# 展示 edge_attr 内容
if "edge_attr" in ckpt:
    edge_attr = ckpt["edge_attr"]
    print(f"\n== edge_attr ==")
    print(f"shape: {edge_attr.shape}")
    # 打印前5条边的全部特征
    print("前5条边的特征:")
    print(edge_attr[:20])
    # 打印每一维的均值和方差，便于理解分布
    print("每一维的均值:", edge_attr.float().mean(dim=0).tolist())
    print("每一维的方差:", edge_attr.float().var(dim=0).tolist())
    # 打印所有空间边（edge_attr[:,9]==1）的特征
    spatial_mask = (edge_attr[:,9] == 1)
    spatial_edges = edge_attr[spatial_mask]
    print(f"\n空间边数量: {spatial_edges.shape[0]}")
    if spatial_edges.shape[0] > 0:
        print("前5条空间边的特征:")
        print(spatial_edges[:5])
        print("空间边每一维的均值:", spatial_edges.float().mean(dim=0).tolist())
        print("空间边每一维的方差:", spatial_edges.float().var(dim=0).tolist())
    else:
        print("没有空间边（edge_attr[:,9]==1）")
else:
    print("未找到 edge_attr 字段")
# 选择模型权重部分
if "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("\n== 权重参数 ==")
for k, v in state_dict.items():
    print(f"{k:60s} shape: {tuple(v.shape)}")
    # 打印部分内容示例
    # print(f"  values: {v.flatten()[:5].tolist()} ...")