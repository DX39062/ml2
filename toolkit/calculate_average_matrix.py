import os
import glob
import scipy.io
import numpy as np

def calculate_average_matrix():
    """
    在当前路径下查找所有 .mat 文件，
    加载其中 116x116 的数据矩阵，
    计算所有矩阵的逐元素平均值，
    并将结果保存为 'FC.csv'。
    """
    
    # 定义预期的数据维度
    expected_shape = (116, 116)
    output_filename = "FC.csv"
    
    # 1. 查找当前路径下所有的 .mat 文件
    mat_files = glob.glob('*.mat')
    
    if not mat_files:
        print("在当前路径下未找到 .mat 文件。")
        return

    print(f"找到了 {len(mat_files)} 个 .mat 文件，开始处理...")
    
    # 2. 初始化一个用于累加的矩阵和计数器
    # 我们需要使用浮点数来存储总和，以确保平均值计算的精度
    total_matrix_sum = np.zeros(expected_shape, dtype=np.float64)
    valid_file_count = 0
    
    # 3. 循环处理每一个 .mat 文件
    for mat_filename in mat_files:
        print(f"--- 正在处理: {mat_filename} ---")
        
        try:
            # 4. 加载 .mat 文件
            data_dict = scipy.io.loadmat(mat_filename)
            
            # 5. 提取数据矩阵
            data_matrix = None
            variable_name = None
            
            for key, value in data_dict.items():
                # 跳过 .mat 文件的元数据
                if key.startswith('__'):
                    continue
                
                # 检查数据是否符合 116x116 的要求
                if isinstance(value, np.ndarray) and value.shape == expected_shape:
                    data_matrix = value
                    variable_name = key
                    print(f"在文件中找到符合 {expected_shape} 维度的变量: '{key}'")
                    # 假设每个文件只包含一个我们想要的矩阵
                    break
                elif isinstance(value, np.ndarray):
                    print(f"找到变量 '{key}'，但维度为 {value.shape}，不符合要求，已跳过。")
                
            # 6. 验证和累加
            if data_matrix is not None:
                # 将当前矩阵加到总和中
                total_matrix_sum += data_matrix
                valid_file_count += 1
                print(f"已将 {mat_filename} (变量 '{variable_name}') 添加到总和。")
            else:
                print(f"未能在 {mat_filename} 中找到符合 {expected_shape} 维度的有效数据。")

        except Exception as e:
            print(f"处理 {mat_filename} 时发生错误: {e}")

    # 7. 检查是否找到了有效文件
    if valid_file_count == 0:
        print("\n--- 处理完成 ---")
        print(f"未能找到任何包含 {expected_shape} 维度数据的有效 .mat 文件。")
        print(f"未生成 {output_filename}。")
        return

    # 8. 计算平均值
    print(f"\n--- 计算平均值 ---")
    print(f"总共处理了 {valid_file_count} 个有效文件。")
    
    # 逐元素相除
    average_matrix = total_matrix_sum / valid_file_count

    # 9. 保存为 .csv 文件
    print(f"正在将平均矩阵保存到 {output_filename}...")
    np.savetxt(output_filename, average_matrix, delimiter=',')
    
    print(f"\n--- 处理完成 ---")
    print(f"成功创建 {output_filename}，其中包含 {valid_file_count} 个矩阵的平均值。")

# --- 运行脚本 ---
if __name__ == "__main__":
    calculate_average_matrix()