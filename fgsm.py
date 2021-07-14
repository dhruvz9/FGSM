import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mobilenet_model = tf.keras.applications.MobileNetV2()
mobilenet_model.trainable = False
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image
def get_label(features):
    return tf.keras.applications.mobilenet_v2.decode_predictions(features, top=1)[0][0]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_src', 
     default = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg', help = 'img src')
args, unknown = parser.parse_known_args()
image_path = tf.keras.utils.get_file('image', args.img_src)
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
print(type(image_raw))
image = preprocess(image)
print(image)
features = mobilenet_model.predict(image)
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_label(features)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()
loss_object = tf.keras.losses.CategoricalCrossentropy()
def adv_image_gen(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = mobilenet_model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad
class_index = features.argmax(axis=-1)
label = tf.one_hot(class_index, features.shape[-1])
label = tf.reshape(label, (1, features.shape[-1]))
perturbations = adv_image_gen(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);
def display_images(image, description):
    _, label, confidence = get_label(mobilenet_model.predict(image))
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
    plt.show()
epsilons = [0, 0.01, 0.15, 0.2]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i])

