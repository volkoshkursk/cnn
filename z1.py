import numpy as np
from test import generate
import progressbar
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from time import *
from PIL import Image

mode = int(input('выберите режим: обчение - 0, оценка - 1, распознать - 2 \n'))
#mode = 2
def encrypt(inp):
	if ord(inp) < 58:
		return ord(inp) - 48
	else:
		return ord(inp) - 55
		
def decrypt(inp):
	if inp < 10:
		return chr(inp + 48)
	else:
		return chr(inp + 55)
		

def picture_load(name,num_of_sym, show = False):
	image = Image.open(name) #Открываем изображение.
	width = image.size[0] #Определяем ширину. 
	height = image.size[1] #Определяем высоту. 	
	pix = image.load() #Выгружаем значения пикселей.
	letters = []
	for t in range(num_of_sym): #дробление на буквы (известно, что их 6)
#		letter = Image.new('1', [(width // num_of_sym), height])
#		draw_let = ImageDraw.Draw(letter)
		letter = np.empty( ((width // num_of_sym), height) , dtype = bool)
		dead_rows = 0
		jn = 0
		for j in range(height): # сие "y"
			num_of_white = 0
			for i in range(t * (width // num_of_sym), t * (width // num_of_sym) + (width // num_of_sym)): # сие "x"
				S = pix[i, j][0] + pix[i, j][1] + pix[i, j][2]
				if (S > (((255) // 2) * 3)):
					letter [i - t * (width // num_of_sym)  ,j - jn] = 1
					num_of_white += 1
				else:
					letter [i - t * (width // num_of_sym) ,j - jn] = 0
			if num_of_white == (width // num_of_sym):
				letter = np.delete(letter, (j -jn), axis=1)
				jn += 1
				del num_of_white
		if jn != 0:
# ----------------выравнивание буквы по вертикали------------
			#letter = np.concatenate(np.ones((int(width//num_of_sym), int(jn//2)), dtype = bool),np.concatenate(letter,np.ones((int(width // num_of_sym), int(jn//2 + jn % 2)), dtype=bool)) )
			letter = np.concatenate((np.ones((int(width//num_of_sym), int(jn//2)), dtype = bool),np.concatenate((letter,np.ones((int(width // num_of_sym),int(jn//2 + jn % 2)), dtype=bool)), axis=1)),axis=1)
		letters.append(letter)
		del dead_rows, letter
	if show:
		f = open('ans.txt', 'w')
		b =''
		for k in letters:
			b += '?' + str(k.shape) + '\n'
			for i in range(k.shape[1]):
				for j in range(k.shape[0]):
						if k[j,i]:
							b += '1'
						else:
							b += '0'
				b += '\n'
			b += '\n============\n'
		f.write(b)
		f.close()
	del image
	return letters


def save(pic,y,mode):
	wr = np.array(pic)
	ywr = np.array(y)
	if mode:
		np.save('save.npy', wr)
		np.save('ans_save.npy', ywr)
	else:
		np.save('save_test.npy', wr)
		np.save('ans_save_test.npy', ywr)
	
def read(name1, name2):
	pic = np.load(name1)
	y = np.load(name2)
	return pic,y

def create(mode):
	y = []
	pic = []
	print('loading training')
	widgets = [progressbar.Percentage(), progressbar.Bar()]
	bar = progressbar.ProgressBar(widgets=widgets, max_value=10000).start()
	for i in range(10000):
		bar.update(i)
		y += generate()
		sleep(0.1)
		pic += picture_load('captcha.png', 6)
	bar.finish()
	#print(y)
	y = [encrypt(x) for x in y]
	#print(y)
	save(pic, y, mode)	
	return np.array(pic),np.array(y)
#Слой свертки, 75 карт признаков, размер ядра свертки: 5х5.
#Слой подвыборки, размер пула 2х2.
#Слой свертки, 100 карт признаков, размер ядра свертки 5х5.
#Слой подвыборки, размер пула 2х5.
#Полносвязный слой, 562 нейронов.
#Полносвязный выходной слой, 36 нейронов, которые соответствуют классам рукописных символов 0 - 9 и букв A - Z.

#picture_load('captcha.png', 6, show = True)
# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Загружаем данные (60000)
try:
	pic,y = read('save.npy', 'ans_save.npy')
except Exception as E:
	pic,y = create(True)
#	if E == FileNotFoundError(2, 'No such file or directory'):
#		print(E)

input_shape = (pic[0].shape[0],pic[0].shape[1] , 1)
# Нормализация данных
pic = pic.astype('float32')
pic = pic.reshape(pic.shape[0], pic[0].shape[0], pic[0].shape[1] , 1)
#pic /= 255

# Преобразуем метки в категории
y = np_utils.to_categorical(y, 36)

# Создаем последовательную модель
model = Sequential()
model.add(Conv2D(75, kernel_size=(5, 5),
 activation='relu',
 input_shape=(pic[0].shape[0],pic[0].shape[1] , 1))) # Слой свертки
model.add(MaxPooling2D(pool_size=(2, 2))) #Слой подвыборки
model.add(Dropout(0.2)) # коэффициент отсева
model.add(Conv2D(111, (3, 3), activation='relu')) # Слой свертки
model.add(MaxPooling2D(pool_size=(2, 5))) # Слой подвыборки
model.add(Dropout(0.2)) # коэффициент отсева
model.add(Flatten()) # выравнивание (трансформация в одномерный вектор (https://keras.io/layers/core/#flatten))
model.add(Dense(562, activation='relu')) # Полносвязный слой
model.add(Dropout(0.5)) # коэффициент отсева
model.add(Dense(36, activation='softmax')) # Полносвязный слой

# Компилируем модель (оптимизатор Адама)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
def predict():
	model.set_weights(np.load('weight.npy'))
	print(model.summary())
	pic = picture_load('captcha_.png', 6, show = True)
	pic = np.array(pic)
	pic = pic.astype('float32')
	pic = pic.reshape(pic.shape[0], pic[0].shape[0], pic[0].shape[1] , 1)
	ans = model.predict(pic, verbose = 1)
	ret = ''
	for i in ans:
		ret += decrypt(np.argmax(i))
	return ret
if not mode:
	# Обучаем сеть
	
	model.fit(pic, y, batch_size=200, epochs=35, validation_split=0.2, verbose=2)

	np.save('weight.npy', model.get_weights())
	f = open('config', 'w')
	f.write(str(model.get_config()))
	f.close()

	# Оцениваем качество обучения сети на тестовых данных
	try:
		pic,y = read('save_test.npy', 'ans_save_test.npy')
	except Exception as e:
		pic,y = create(False)
	# Нормализация данных
	pic = pic.astype('float32')
	pic = pic.reshape(pic.shape[0], pic[0].shape[0], pic[0].shape[1] , 1)
	#pic /= 255
	
	# Преобразуем метки в категории
	y = np_utils.to_categorical(y, 36)
	scores = model.evaluate(pic,y, verbose=1)
	print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
	

	

#	model.save_weights('weight')
	
elif mode == 1:
	model.set_weights(np.load('weight.npy'))
#	pic = np.array(picture_load('captcha.png', 6)).astype('float32')
	print(model.summary())
#	print(pic.shape)
#	print(model.predict(pic.reshape(pic.shape[0], pic.shape[1],pic.shape[2]  , 1)))
	# Оцениваем качество обучения сети на тестовых данных
	try:
		pic,y = read('save_test.npy', 'ans_save_test.npy')
	except Exception as e:
		pic,y = create(False)
	# Нормализация данных
	pic = pic.astype('float32')
	pic = pic.reshape(pic.shape[0], pic[0].shape[0], pic[0].shape[1] , 1)
	#pic /= 255
	
	# Преобразуем метки в категории
	y = np_utils.to_categorical(y, 36)
	scores = model.evaluate(pic,y, verbose=1)
	print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

else:
	print(predict())