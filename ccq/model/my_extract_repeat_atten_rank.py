import torch

def select_indices(tensor, k):
    """
    Extract the indices of n0 elements sorted in descending order between the first -1 and the second -1 in each row. Then, extract the indices of n0 elements sorted in descending order each time after the second -1, until k indices are collected in total.

    Parameters:
    tensor (torch.Tensor): Input tensor.
    k (int): Total number of indices to return.

    Returns:
    torch.Tensor: Tensor of indices for each row, with shape (num_rows, k).
    """
    selected_indices = []

    for row in tensor:
        row_indices = []


        first_neg1_idx = (row == -1).nonzero(as_tuple=True)[0][0]
        second_neg1_idx = (row == -1).nonzero(as_tuple=True)[0][1]


        middle_part_indices = torch.arange(first_neg1_idx + 1, second_neg1_idx, device=row.device)
        n0 = len(middle_part_indices)


        first_part_sorted_indices = middle_part_indices[
            row[first_neg1_idx + 1:second_neg1_idx].argsort(descending=True)]
        row_indices.extend(first_part_sorted_indices.tolist())


        current_idx = second_neg1_idx + 1
        while len(row_indices) < k:
            # 取接下来的 n0 个元素的下标
            next_part_indices = torch.arange(current_idx, min(current_idx + n0, len(row)), device=row.device)
            sorted_indices = next_part_indices[row[next_part_indices].argsort(descending=True)]
            row_indices.extend(sorted_indices.tolist())
            current_idx += n0


        row_indices = row_indices[:k]
        selected_indices.append(row_indices)

    #  torch.Tensor
    return torch.tensor(selected_indices, device=tensor.device)

if __name__ == '__main__':
    # 示例输入张量
    tx = torch.tensor([
        [-1., 1., 2., 3., -1., 1., 2., 3., 1., 2., 3., 1., 2.],
        [-1., 4., 5., 6., -1., 4., 5., 6., 4., 5., 6., 4., 5.],
        [-1., 7., 8., 9., -1., 7., 8., 9., 7., 8., 9., 7., 8.],
        [-1., 10., 11., -1., 10., 11., 10., 11., 10., 11., 10., 11., 10.],
        [-1., 12., 13., 6., 8., 9., -1., 12., 13., 6., 8., 9., 12.],
        [-1., 12., 2., 1., 8., 3., -1., 12., 2., 1., 8., 3., 12.],
        [-1., 41., 11., -1., 41., 11., 41., 11., 41., 11., 41., 11., 41.],
        [-1., 4., 11., -1., 4., 11., 4., 11., 4., 11., 4., 11., 4.],
        [-1., 4., 11., 1., -1, 4., 11., 1.,  4., 11., 1., 4.,  11.]
    ]).to('cuda')

    k = 8

    # index
    indices = select_indices(tx, k)

    # result
    print(indices)
