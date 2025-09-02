import torch
# -*- coding: utf-8 -*-

def replace_vectors_in_matrix(New_base_features, dynamicLinearProjection_group, result):
    """
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(New_base_features):
        New_base_features = torch.tensor(New_base_features)
    if not torch.is_tensor(dynamicLinearProjection_group):
        dynamicLinearProjection_group = torch.tensor(dynamicLinearProjection_group)

    # Check input shapes and compatibility
    if dynamicLinearProjection_group.shape[1] != New_base_features.shape[1]:
        raise ValueError(
            f"Vector dimension {dynamicLinearProjection_group.shape[1]} does not match target matrix column count {New_base_features.shape[1]}")
    if dynamicLinearProjection_group.shape[0] != len(result):
        raise ValueError(f"Number of vectors {dynamicLinearProjection_group.shape[0]} does not match number of positions {len(result)}")

    # Replace corresponding rows in New_base_features according to indices in result
    for i, pos in enumerate(result):
        if pos >= New_base_features.shape[0]:
            raise ValueError(f"Position index {pos} exceeds matrix row count {New_base_features.shape[0]}")
        New_base_features[pos] = dynamicLinearProjection_group[i]

    return New_base_features


if __name__ == '__main__':

    # Example usage
    # Initialize data
    New_base_features = torch.zeros(128, 1536)  # Target matrix of shape 128×1536
    dynamicLinearProjection_group = torch.tensor([
        [3.9922, -2.0234, 2.0352] + [0] * 1533,  # First row vector, padded to 1536 dimensions
        [0.5679, 3.5332, 2.9766] + [0] * 1533  # Second row vector, padded to 1536 dimensions
    ])  # Tensor of shape 2×1536
    result = [63, 91]  # Replacement positions

    # Call the function
    updated_matrix = replace_vectors_in_matrix(New_base_features, dynamicLinearProjection_group, result)

    # Verify results
    print(updated_matrix[63][:3])  # Should display [3.9922, -2.0234, 2.0352]
    print(updated_matrix[91][:3])  # Should display [0.5679, 3.5332, 2.9766]
