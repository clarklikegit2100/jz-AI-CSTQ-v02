datasets = [
    ("Fluo-C2DL-Huh7",   672,  37.1, "DONE"),
    ("DIC-C2DH-HeLa",   1968,  None, ""),
    ("Fluo-N2DH-GOWT1", 2160,  None, ""),
    ("Fluo-N2DH-SIM+",  2368,  None, ""),
    ("PhC-C2DH-U373",   2712,  None, ""),
    ("PhC-C2DL-PSC",    1192,  None, "4 seqs"),
]
MS = 985  # actual stable ms/batch
print()
print("=" * 65)
print("  修正时间估算（实测 985ms/batch）")
print("=" * 65)
col1, col2, col3, col4 = "数据集", "样本", "估算/min", "状态"
print(f"  {col1:<22} {col2:>6} {col3:>10} {col4:>10}")
print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10}")
total_rem = 0
for name, n, actual, note in datasets:
    est = (n * MS / 1000) / 60
    if actual:
        print(f"  {name:<22} {n:>6}  {actual:>8.1f}  {'OK ' + note:>10}")
    else:
        total_rem += est
        print(f"  {name:<22} {n:>6}  {est:>8.1f}  {note:>10}")
print("=" * 65)
print(f"  剩余 5 个数据集合计约 {total_rem:.0f} min = {total_rem/60:.1f} h")
print()
