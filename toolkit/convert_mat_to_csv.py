import os
import glob
import scipy.io
import numpy as np

def convert_mat_files():
    """
    在当前路径下查找所有 .mat 文件，
    加载它们，并将数据矩阵保存为 .csv 文件。
    
    输入命名: ROISignals_129_S_6228_20180220.mat
    输出命名: 129_S_6228_20180220.csv
    """
    
    # 1. 查找当前路径下所有的 .mat 文件
    mat_files = glob.glob('*.mat')
    
    if not mat_files:
        print("在当前路径下未找到 .mat 文件。")
        return

    print(f"找到了 {len(mat_files)} 个 .mat 文件，开始处理...")
    
    processed_count = 0
    
    # 2. 循环处理每一个 .mat 文件
    for mat_filename in mat_files:
        print(f"\n--- 正在处理: {mat_filename} ---")
        
        try:
            # 3. 生成输出文件名
            # 'ROISignals_129_S_6228_20180220.mat' -> 'ROISignals_129_S_6228_20180220'
            base_name = os.path.splitext(mat_filename)[0]
            
            # 按第一个 '_' 分割
            parts = base_name.split('_', 1)
            
            # 检查文件名是否符合规范（至少包含一个 '_'）
            if len(parts) < 2:
                print(f"文件名 {mat_filename} 不包含 '_'，无法按规则重命名，已跳过。")
                continue
                
            # '129_S_6228_20180220'
            new_base_name = parts[1]
            # '129_S_6228_20180220.csv'
            output_filename = new_base_name + ".csv"
            
            # 4. 加载 .mat 文件
            print(f"加载 {mat_filename}...")
            # .mat 文件加载后是一个字典
            data_dict = scipy.io.loadmat(mat_filename)
            
            # 5. 提取数据矩阵
            # 我们需要找到真正的数据变量，而不是元数据（如 '__header__'）
            data_matrix = None
            variable_name = None
            
            for key, value in data_dict.items():
                if not key.startswith('__'):
                    # 假设文件中的第一个非元数据变量就是我们要的数据矩阵
                    data_matrix = value
                    variable_name = key
                    print(f"在 .mat 文件中找到数据变量: '{key}'")
                    break
            
            # 6. 检查数据
            if data_matrix is None:
                print(f"未能在 {mat_filename} 中找到有效的数据变量。")
                continue
                
            # 确保它是一个二维矩阵
            if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2:
                print(f"变量 '{variable_name}' 不是一个二维矩阵 (shape: {data_matrix.shape})，已跳过。")
                continue

            #您提到数据是 140*116，我们可以加一个检查（如果需要，可以取消注释）
            if data_matrix.shape != (140, 116):
                print(f"警告: 数据维度为 {data_matrix.shape}，而非预期的 (140, 116)。")

            # 7. 将数据保存为 .csv 文件
            print(f"正在保存数据到 {output_filename}...")
            # 使用 numpy.savetxt 可以方便地保存2D数组，并指定分隔符为逗号
            np.savetxt(output_filename, data_matrix, delimiter=',')
            
            print(f"成功转换 {mat_filename} -> {output_filename}")
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {mat_filename} 时发生错误: {e}")

    print(f"\n--- 处理完成 ---")
    print(f"总共成功处理了 {processed_count} / {len(mat_files)} 个文件。")

# --- 运行脚本 ---
if __name__ == "__main__":
    convert_mat_files()