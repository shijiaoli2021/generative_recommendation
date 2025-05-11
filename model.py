import paddle.nn as nn
import paddle

class Modu(nn.Layer):

    def __init__(self):
        super(Modu, self).__init__()
        self.embedding = nn.Embedding(7000, 256)



if __name__ == '__main__':
    m = Modu()
    x = paddle.rand(shape=(32, 128))
    codebook = paddle.rand(shape=(3, 128))
    out = (x**2).sum(axis=1, keepdim=True) + (codebook.T ** 2).sum(axis=0, keepdim=True) - 2 * x @ codebook.T
    print(out.shape)
    a = paddle.rand(shape=(32, 1))
    b = paddle.rand(shape=(1, 3))
    print((a+b).shape)