class TensorSaver:
    def __init__(self):
        self.saved_tensors = []

    def add_tensor(self, tensor):
        self.saved_tensors.append(tensor)

    def print_tensors(self):
        print("Saved tensors:", self.saved_tensors)

saver = TensorSaver()

current_epoch=0

flag_num=-1
count_num=0

pidstatistics = 0