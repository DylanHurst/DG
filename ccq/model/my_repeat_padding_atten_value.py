import torch

def transform_matrix(matrix):
    """
    Format each row of the input matrix such that the start and end of each row are -1, and the middle part is filled by repetition.

    Parameters:
    matrix (torch.Tensor): Input matrix, which contains -1 at the start and end, as well as the middle part and zero padding.

    Returns:
    torch.Tensor: The formatted matrix, which retains the original row length.
    """
    transformed_matrix = []

    for row in matrix:
        # 找到首尾的 -1 的位置
        start_index = torch.where(row == -1)[0][0].item()
        end_index = torch.where(row == -1)[0][1].item()

        # 获取需要重复的中间部分
        middle_part = row[start_index + 1:end_index]

        # 计算目标长度
        target_length = len(row)

        # 构建新行：[首部 -1] + [中间部分] + [尾部 -1]
        new_row = torch.cat((row[start_index:start_index+1], middle_part, row[end_index:end_index+1]))

        # 持续重复中间部分填充剩余长度，直到达到目标长度
        while len(new_row) < target_length:
            new_row = torch.cat((new_row, middle_part))

        # 修剪到确切长度
        new_row = new_row[:target_length]

        # 将新行加入结果矩阵
        transformed_matrix.append(new_row)

    # 转换为 torch 张量形式
    return torch.stack(transformed_matrix)


def main():
    # 原始矩阵
    matrix = torch.tensor([
        [-1, 1, 2, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 4, 5, 6, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 7, 8, 9, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 10, 11, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 12, 13, 6, 8, 9, -1, 0, 0, 0, 0, 0, 0],
        [-1, 12, 2, 1, 8, 3, -1, 0, 0, 0, 0, 0, 0],
        [-1, 4, 11, -1, 8, 3, 6, 0, 0, 0, 0, 0, 0],
        [-1, 4, 11, -1, 8, 3, 6, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)

    # 调用 transform_matrix 函数并获取结果
    transformed_matrix = transform_matrix(matrix)

    # 输出结果
    print(transformed_matrix)


if __name__ == "__main__":
    main()
