"""
从已保存的Excel文件生成四目标平行坐标图
"""
import sys
from pathlib import Path

# 添加当前目录到路径
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# 导入函数
from multi_objective_optimization_nsga2 import plot_parallel_coordinates_from_file

# Excel文件路径
excel_file = CURRENT_DIR / "SAVE" / "pareto_optimal_solutions.xlsx"

if __name__ == "__main__":
    if excel_file.exists():
        print(f"从Excel文件生成平行坐标图: {excel_file}")
        best_idx, best_values = plot_parallel_coordinates_from_file(
            str(excel_file),
            save_path=str(CURRENT_DIR / "SAVE" / "pareto_front_parallel_coordinates.png")
        )
        print(f"\n最佳平衡解索引: {best_idx}")
        print(f"最佳平衡解数值:")
        print(f"  Price: {best_values[0]:.2f} $/m³")
        print(f"  Strength: {best_values[1]:.2f} MPa")
        print(f"  CO₂: {best_values[2]:.2f} kg/m³")
        print(f"  Ductility: {best_values[3]:.4f}")
    else:
        print(f"错误: Excel文件不存在: {excel_file}")
        print("请先运行多目标优化生成 pareto_optimal_solutions.xlsx")

