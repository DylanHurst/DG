class TensorSaver:
    def __init__(self):
        # 定义公有列表变量
        self.saved_tensors = []

    # 函数 1：向 saved_tensors 中添加数据
    def add_tensor(self, tensor):
        self.saved_tensors.append(tensor)

    # 函数 2：打印 saved_tensors 的内容
    def print_tensors(self):
        print("Saved tensors:", self.saved_tensors)

# 定义全局实例
saver = TensorSaver()

current_epoch=0

flag_num=-1
count_num=0

pidstatistics = 0