import random
import string
from claptcha import Claptcha
from PIL import Image

def randomString():
	rndLetters = (random.choice(string.ascii_uppercase) for _ in range(6))
	return "".join(rndLetters)

def randomNum():
	rndLetters = (random.choice(string.digits) for _ in range(6))
	return "".join(rndLetters)

def randomMix():
	rndLetters = (random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
	return "".join(rndLetters)
def generate(name = 'captcha.png'):
	# Initialize Claptcha object with random text
	c = Claptcha(randomMix, "Roboto-BlackItalic.ttf",
				 resample=Image.BICUBIC, noise=0.3)
	text, _ = c.write(name)
	return (text)  # string printed into captcha.png