from PIL import Image
from mindspore import Tensor, nn

from clip_model import load, tokenize



model, preprocess = load("ViT_B_16.ckpt", device="GPU")

image = Tensor(preprocess(Image.open("CAT.jpg")))
text = tokenize(["a diagram", "Onoclea sensibilis", "a cat", "Blastodacna bicristatella", "Bluethroat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = nn.Softmax(axis=-1)(logits_per_image).numpy()

print("Label probs:", probs)