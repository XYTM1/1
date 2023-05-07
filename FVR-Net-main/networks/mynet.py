# 导入一些必要的库
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Dense, Lambda, Layer
from tensorflow.python.keras import backend as K


# 定义一个类DUF，继承自tf.keras.Model，作为VSR-DUF模型的主体
class DUF(tf.keras.Model):
    def __init__(self, t_in=7, t_out=7, scale=4, r=4, n0=256, n1=64):
        # 在初始化函数中，定义一些超参数，并创建一些卷积层和全连接层作为模型的组件
        super(DUF, self).__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.scale = scale
        self.r = r
        self.n0 = n0
        self.n1 = n1

        # 特征提取卷积层，输入为低分辨率视频序列，输出为特征张量
        self.conv1 = Conv2D(n0, 3, padding='same', activation='relu', name='conv1')

        # 循环神经网络（RNN）层，输入为特征张量，输出为每一帧对应的动态上采样滤波器（DUF）和残差（RES）
        self.rnn1 = ConvLSTM(n0, 3, padding='same', return_sequences=True, name='rnn1')
        self.rnn2 = ConvLSTM(n0, 3, padding='same', return_sequences=True, name='rnn2')
        self.conv2 = Conv2D(n0 * (t_out * r * r + 1), 1, padding='same', name='conv2')

        # 上采样卷积层，输入为低分辨率帧和DUF，输出为上采样后的高分辨率帧
        self.conv3 = Conv2D(n1 * r * r * t_out, 3, padding='same', name='conv3')
        self.pixel_shuffle = Lambda(
            lambda x: tf.nn.depth_to_space(x, r),
            name='pixel_shuffle'
        )

    def call(self, inputs):
        # 在call函数中，实现模型的前向传播逻辑

        # 将输入的低分辨率视频序列拼接成一个张量，并通过一个卷积层提取特征
        x = tf.concat(inputs, axis=-1)
        x = self.conv1(x)

        # 使用一个循环神经网络（RNN）对特征进行时序建模，并输出每一帧对应的动态上采样滤波器（DUF）和残差（RES）
        x = tf.expand_dims(x, axis=0)
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = tf.squeeze(x)
        x = self.conv2(x)

        # 将DUF和RES拆分成两个张量，并调整形状以便后续处理
        duf = x[:, :, :, :self.t_out * self.r * self.r]
        res = x[:, :, :, self.t_out * self.r * self.r:]

        duf_shape = K.shape(duf)
        duf_reshape_shape = [duf_shape[0], duf_shape[1], duf_shape[2],
                             self.t_out * self.r * self.r // (self.r * 2),
                             self.r * 2]

        res_shape =duf = x[:, :, :, :self.t_out * self.r * self.r]
        res = x[:, :, :, self.t_out * self.r * self.r:]

        duf_shape = K.shape(duf)
        duf_reshape_shape = [duf_shape[0], duf_shape[1], duf_shape[2],
                             self.t_out * self.r * self.r // (self.r * 2),
                             self.r * 2]

        res_shape = K.shape(res)
        res_reshape_shape = [res_shape[0], res_shape[1], res_shape[2],
                             self.t_out, 1]

        # 将DUF和RES重塑为5维张量，并沿着第四维拆分成t_out个张量
        duf = tf.reshape(duf, duf_reshape_shape)
        res = tf.reshape(res, res_reshape_shape)
        dufs = tf.split(duf, self.t_out, axis=3)
        ress = tf.split(res, self.t_out, axis=3)

        # 使用DUF对输入的低分辨率帧进行上采样，并加上RES得到高分辨率帧
        outputs = []
        for i in range(self.t_out):
            # 取出第i个DUF和RES，并将DUF转置为合适的形状
            duf_i = dufs[i]
            res_i = ress[i]
            duf_i = tf.transpose(duf_i, perm=[0, 1, 2, 4, 3])

            # 取出第i个低分辨率帧，并通过一个卷积层得到n1*r*r个通道
            lr_i = inputs[i + (self.t_in - self.t_out) // 2]
            lr_i = self.conv3(lr_i)

            # 将低分辨率帧和DUF相乘，并沿着最后一个维度求和，得到上采样后的高分辨率帧
            hr_i = tf.multiply(lr_i, duf_i)
            hr_i = tf.reduce_sum(hr_i, axis=-1)

            # 将高分辨率帧和残差相加，并使用像素重排操作得到最终的高分辨率帧
            hr_i += res_i
            hr_i = self.pixel_shuffle(hr_i)

            # 将高分辨率帧添加到输出列表中
            outputs.append(hr_i)

        # 将高分辨率帧拆分成一个序列，并返回作为模型的输出
        outputs = tf.stack(outputs, axis=-1)
        outputs = tf.split(outputs, self.t_out, axis=-1)
        return outputs

# 定义一个函数create_model，用于根据给定的参数创建一个DUF模型，并加载预训练的权重
def create_model(t_in=7, t_out=7, scale=4):
    model = DUF(t_in=t_in, t_out=t_out, scale=scale)
    model.load_weights('weights/duf_{}x.h5'.format(scale))
    return model