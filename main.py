import torch

class A:
    def __init__(self):
        self.model = None
        
class B(A):
    def __init__(self):
        super().__init__()
        del self.model

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    b = B()
    b.model
