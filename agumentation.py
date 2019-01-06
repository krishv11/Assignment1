from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

datagen = ImageDataGenerator(
    shear_range= 0.2,
    zoom_range= 0.1,
    horizontal_flip= True
)
#printing the augmentation of network
img = load_img('D:\\Documents\\Assignment1\\Lab21\\venv\\nonfaces\\1.png')
x = img_to_array((img))
x = x.reshape((1,)+x.shape)
i = 0

for batch in datagen.flow(x,batch_size=1,
    save_to_dir='Grey_Train',save_format='png'):
    i += 1
    if i > 10:
        break
