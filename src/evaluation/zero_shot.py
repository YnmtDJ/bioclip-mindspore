from PIL import Image
from mindspore import Tensor, nn
import mindspore
from tqdm import tqdm
from ..training.precision import get_autocast
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from ..training.logger import setup_logging

from ..clip_mindspore.clip_model import load, tokenize
from ..clip_mindspore.func import get_cast_dtype, DatasetFromFile

import logging

from .params import parse_args
from .utils import init_device, random_seed

import sys
import os
import datetime

# model, preprocess = load("../../BIOCLIP.ckpt", device="GPU")
#
# image = Tensor(preprocess(Image.open("D:/Users/user\PycharmProjects\clip-mindspore\data\insects_mini_1k\images/0a028522-0177-4087-86fa-9d7391f69b6a.jpg")))
# text = tokenize(["a diagram", "Onoclea sensibilis", "a cat", "Blastodacna bicristatella", "Bluethroat"])
#
# image_features = model.encode_image(image)
# text_features = model.encode_text(text)
#
# logits_per_image, logits_per_text = model(image, text)
# probs = nn.Softmax(axis=-1)(logits_per_image).numpy()
#
# print("Label probs:", probs)


def get_dataloader(dataset):
    return mindspore.dataset.GeneratorDataset(
        source=dataset,
        column_names=["img", "label"]
    )


def zero_shot_classifier(model, classnames, templates, args):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template(classname) for template in templates]  # format with class
        texts = tokenize(texts)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = mindspore.ops.mean(mindspore.ops.L2Normalize(axis=-1)(class_embeddings), axis=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = mindspore.ops.stack(zeroshot_weights, axis=1)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return dict([
        (k, float(correct[:k].reshape(-1).float().sum(0, keepdim=True).numpy()))
        for k in topk
    ])


def run(model, classifier, dataloader, args):
    # get_autocast会将model设置为自动混合精度
    # model = get_autocast(args.precision, model)
    cast_dtype = get_cast_dtype(args.precision)
    n = 0.0
    topk = dict()
    for i in (1, min(len(dataloader.source.classes), 3), min(len(dataloader.source.classes), 5)):
        topk[i] = 0.0
    for images, target in tqdm(dataloader.batch(args.batch_size), unit_scale=args.batch_size):
        # images.shape: torch.Size([batch_size, 3 rgb channels, image_height, image_width])
        images = images.squeeze(axis=1)  # batch load need to squeeze dimension
        if cast_dtype is not None:
            images = images.to(dtype=cast_dtype)

        # with autocast():
        # predict
        image_features = model.encode_image(images)
        image_features = mindspore.ops.L2Normalize(axis=-1)(image_features)
        # logits = 100.0 * image_features @ classifier
        logits = model.logit_scale.exp() * image_features @ classifier

        # measure accuracy
        acc = accuracy(logits, target, topk=topk.keys())
        for k, v in acc.items():
            topk[k] += v
        n += images.shape[0]

    for k, v in acc.items():
        topk[k] /= n
    return topk


def zero_shot_eval(model, data, args):
    results = {}

    logging.info("Starting zero-shot.")

    for split in data:
        logging.info("Building zero-shot %s classifier.", split)
        classnames = data[split].source.classes

        classifier = zero_shot_classifier(
            model, classnames, openai_imagenet_template, args
        )

        topk = run(model, classifier, data[split], args)

        for k, v in topk.items():
            results[f"{split}-top{k}"] = v

        logging.info("Finished zero-shot %s with total %d classes.", split, len(data[split].source.classes))

    logging.info("Finished zero-shot.")

    return results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = init_device(args)

    args.save_logs = args.logs and args.logs.lower() != "none"

    # get the name of the experiments
    if args.save_logs and args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot",
            ]
        )
    if args.save_logs is None:
        args.log_path = None
    else:
        log_base_path = os.path.join(args.logs, args.name)
        args.log_path = None
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if (
            isinstance(args.force_image_size, (tuple, list))
            and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    mindspore.set_deterministic(True)
    model, preprocess = load(args.pretrained, device=device)

    random_seed(args.seed, args.rank)

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    if args.save_logs is None:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    else:
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # initialize datasets
    dataset = DatasetFromFile(args.data_root, args.label_filename, transform=preprocess, classes=args.text_type)
    data = {
        "val-unseen": get_dataloader(
            dataset
        ),
    }

    metrics = zero_shot_eval(model, data, args)

    logging.info("Results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value * 100:.2f}")
    logging.info("Done.")
