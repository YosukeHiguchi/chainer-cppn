import sys
import math

import argparse
import numpy as np
from PIL import Image

import chainer

import net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Tanh', choices=['Tanh', 'Tanh_BN', 'Softplus', 'Relu'])
    parser.add_argument('--func', type=str, default='round', choices=['round', 'heart', 'ellipse', 'sin'])
    parser.add_argument('--n_unit', type=int, default=12)
    parser.add_argument('--n_depth', type=int, default=12)
    parser.add_argument('--cdim', type=int, default=1, choices=[1, 3])
    parser.add_argument('--zdim', type=int, default=10)
    parser.add_argument('--scale', type=int, default=10)
    parser.add_argument('--im_size', type=int, default=1024)
    args = parser.parse_args()

    gen = net.Generator(args.n_unit, args.n_depth, args.cdim, args.mode)

    x_range = args.scale * np.linspace(-1.0, 1.0, args.im_size)
    y_range = args.scale * np.linspace(-1.0, 1.0, args.im_size)
    z = args.scale * np.random.uniform(-1.0, 1.0, (1, args.zdim)).astype(np.float32)

    img = np.empty((0, args.im_size, args.cdim), dtype=np.float32)
    batchsize = 256
    with chainer.using_config('train', False):
        for y in y_range:
            px = np.empty((0, args.cdim), dtype=np.float32)
            for i in range(0, args.im_size, batchsize):
                batch = np.empty((0, 3 + args.zdim), dtype=np.float32)
                for x in x_range[i: min(i + batchsize, args.im_size)]:
                    vec = np.array([x, y, func(args.func)(x, y)], dtype=np.float32).reshape(1, 3)
                    batch = np.append(batch, np.append(vec, z, axis=1), axis=0)
                out = gen(batch)
                px = np.r_[px, out.data]
            img = np.append(img, px[np.newaxis, :, :], axis=0)

    img = img * 255.0
    # img = img * 150.0 + 100.0
    # img[:, :, 0] = img[:, :, 0] * 0
    # img[:, :, 1] = img[:, :, 1] * 0
    # img[:, :, 2] = img[:, :, 2] * 0

    if args.cdim == 1:
        Image.fromarray(img[:, :, -1].astype(np.uint8)).save(
            'result/{}_u{}_d{}_z{}_s{}.png'.format(args.mode, args.n_unit, args.n_depth, args.zdim, args.scale))
    elif args.cdim == 3:
        Image.fromarray(img.astype(np.uint8), 'RGB').save(
            'result/{}_u{}_d{}_z{}_s{}.png'.format(args.mode, args.n_unit, args.n_depth, args.zdim, args.scale))

def func(name):
    func_dict = {
        'round': lambda x, y: math.sqrt(x**2 + y**2),
        'heart': lambda x, y: x**2 + (y - (x**2)**(1 / 3))**2,
        'ellipse': lambda x, y, a=10, b=3: math.sqrt(x**2 * a + y**2 * b),
        'sin': lambda x, y: math.sin((x + y) * mat.pi)
    }

    return func_dict[name]

if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    main()
