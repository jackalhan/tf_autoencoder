import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', default='./text_training',
        help='Output directory for model and training stats.')
    parser.add_argument(
        '--data_dir', default='./text_data',
        help='Directory to download the data to.')
    parser.add_argument('--model', default='convolutional')
    parser.add_argument('--number_of_filters', default="16,8,8")
    parser.add_argument('--dense_layers', default="1:1024,2:512", type=str)
    parser.add_argument('--number_of_tokens', default=144, type=int)
    parser.add_argument('--is_l2_normed', default=True, type=str2bool)
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size (default: 256)')
    parser.add_argument(
        '--noise_factor', type=float, default=0.5,
        help='Amount of noise to add to input (default: 0)')
    parser.add_argument(
        '--dropout', type=float, default=0.5,
        help='The probability that each element is kept in dropout layers (default: 1)')
    parser.add_argument(
        '--loss', type=str, default="custom_distance_loss")
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate (default: 0.001)')
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs to perform for training (default: 50)')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='Amount of weight decay to apply (default: 1e-5)')
    parser.add_argument(
        '--save_images',
        help='Path to directory to store intermediate reconstructed images (default: disabled)')

    parser.add_argument(
        '--images', type=int, default=10,
        help='Number of test images to reconstruct (default: 10)')
    parser.add_argument(
        '--what', choices=['reconstruction', 'embedding'],
        default='embedding',
        help='Whether to display reconstructed images or '
             'create checkpoint with encoder output to visualize '
             'in TensorBoard.')

    return parser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_number_of_filters(number_of_filters_as_arg):
    return [int(filter.strip()) for filter in number_of_filters_as_arg.split(',')]