import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=65, type=int)
    parser.add_argument('--model', default='MSBot', type=str, choices=['Concat', 'MSBot', 'MULSA'])
    parser.add_argument('--model_dir', type=str,
                        help='path where to save or load model')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='epochs to warmup')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='')
    parser.add_argument('--seq_len', default=200, type=int,
                        help='Default action sequence length in MS-Bot.')
    parser.add_argument('--pour_setting', default='init', type=str, choices=['init', 'target'],
                        help='Pouring setting: different init weight or target weight.')
    parser.add_argument('--inference_weight', default=40, type=int,
                        help='Inference pouring setting.')

    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    parser.add_argument(
        "--blur_p", type=float, default=0.25, help="Default random attention blur probability."
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Default stage inject weight beta."
    )
    parser.add_argument(
        "--gamma", type=int, default=15, help="Default soft constraint range near the stage boundaries."
    )
    parser.add_argument(
        "--penalty_intensity", type=float, default=5.0, help="Default score penalty intensity lambda."
    )

    return parser