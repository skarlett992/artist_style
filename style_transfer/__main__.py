import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from style_transfer.learn import StyleTransfer
import numpy as np

def main():
    parser = ArgumentParser(description=("Стизирование картины из контента "
                            "в пример стиля"),
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('content', metavar='<content path>',
                        help='Картин для выполнения стилизации')
    parser.add_argument('style', metavar='<style path>', help='Пример стиля в виде картины')
    parser.add_argument('--artwork', metavar='<path>',
                        default='output/artwork.png',
                        help='Место для сохранения результата <path>.')
    parser.add_argument('--init_img', metavar='<path>', 
                        help=('Initialize artwork with image at <path> '
                        'Другая картинка в качестве контента (ссылка)'))
    parser.add_argument('--init_random', action='store_true',
                        help=('Рандомная картинка из директории'))
    parser.add_argument('--area', metavar='<int>', default=512, type=int, 
                        help=("Разрешение (high resolution"))
    parser.add_argument('--iter', metavar='<int>', default=500, type=int, 
                        help='Количество итераций (чем выше, тем более точно к стилизации')
    parser.add_argument('--lr', metavar='<float>', default=1, type=float, 
                        help='Шаг обучения в оптимайзере')
    parser.add_argument('--content_weight', metavar='<int>', default=1,
                        type=int, help='Вес исходной картинки')
    parser.add_argument('--style_weight', metavar='<int>', default=10,
                        type=int, help='Вес картинки, которую мы используем для стиля')
    parser.add_argument('--coef_style_w', metavar='<int>', default=1,
                        type=int, help='Коэффициент веса картинки, которую мы используем для стиля')
    parser.add_argument('--content_weights', metavar='<str>',
                        default="{'relu_4_2':1}",
                        help=('вес контентной картинки для функции потерь каждого слоя'))
    parser.add_argument('--style_weights', metavar='<str>',
                        default=("{'relu_1_1':1,'relu_2_1':1,"
                        "'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                        help=('Вес функции потерь для картинки стиля для каждого слоя нейронки'))
    parser.add_argument('--avg_pool', action='store_true',
                        help='Replace max-pooling by average-pooling.')
    parser.add_argument('--no_feature_norm', action='store_false',
                        help=("Don't divide each style_weight by the square "
                        "of the number of feature maps in the corresponding "
                        "layer."))
    parser.add_argument('--preserve_color', choices=['content','style','none'],
                        default='content', help=("Если 'style', поменять цвет в стиль "
                        "Если 'none', то цета смешиваются"))
    parser.add_argument('--weights', choices=['original','normalized'],
                        default='original', help=("Вес VGG19"
                        "Также вес VGG19 'original' либо 'normalized' нормализованный."))
    parser.add_argument('--device', choices=['cpu','cuda','auto'],
                        default='auto', help=("Если проблема с gpu," 
                        "то можно включить режим cpu, но каждый запрос "
                         "будет обрабатываться в 20-30 раз дольше."))
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision for training.')
    parser.add_argument('--use_adam', action='store_true',
                        help='Use Adam instead of LBFGS optimizer.')
    parser.add_argument('--optim_cpu', action='store_true',
                        help=('Optimize artwork on CPU to move some data from'
                        ' GPU memory to working memory.'))
    parser.add_argument('--quality', metavar='<int>', default=95, type=int,
                        help=('JPEG image quality of artwork, on a scale '
                        'from 1 to 95.'))
    parser.add_argument('--logging', metavar='<int>', default=50, type=int,
                        help=('Number of iterations between logs. '
                        'If 0, no logs.'))
    parser.add_argument('--seed', metavar='<int>', default='random',
                        help='Количество рандомных генераций')
    args = parser.parse_args()

    if args.seed != 'random':
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
         torch.manual_seed(int(args.seed))

    style_transfer = StyleTransfer(lr=args.lr,
                                   content_weight=args.content_weight,
                                   style_weight=args.style_weight,
                                   content_weights=args.content_weights,
                                   style_weights=args.style_weights,
                                   coef_style_w=args.coef_style_w,
                                   avg_pool=args.avg_pool,
                                   feature_norm=args.no_feature_norm,
                                   weights=args.weights,
                                   preserve_color=
                                   args.preserve_color.replace('none',''),
                                   device=args.device,
                                   use_amp=args.use_amp,
                                   adam=args.use_adam,
                                   optim_cpu=args.optim_cpu, 
                                   logging=args.logging)

    init_img = Image.open(args.init_img) if args.init_img else None
    with Image.open(args.content) as content, Image.open(args.style) as style:
        artwork = style_transfer(content, style,
                                 area=args.area,
                                 init_random=args.init_random,
                                 init_img=init_img,
                                 iter=args.iter)
    artwork.save(args.artwork, quality=args.quality)
    artwork.close()
    if init_img:
        init_img.close()
 
if __name__ == '__main__':
    main()
