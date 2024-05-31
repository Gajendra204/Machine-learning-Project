import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os

# Load the model
model = load_model('FV.h5')

# Define the labels and categories
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


# Function to fetch calories
def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't fetch the Calories")
        print(e)


# Function to process image and predict
def processed_img(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224, 3))
    except Exception as e:
        st.error("Error loading image: " + str(e))
        return None
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    res = labels[int(y_class)]
    return res.capitalize()


# Main function to run the Streamlit app
def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)

        # Create the upload_images directory if it doesn't exist
        if not os.path.exists('./upload_images'):
            os.makedirs('./upload_images')

        save_image_path = os.path.join('./upload_images', img_file.name)

        # Save the uploaded image to the specified path
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Verify if the image has been saved correctly
        if os.path.exists(save_image_path):
            result = processed_img(save_image_path)
            if result:
                if result in vegetables:
                    st.info('**Category : Vegetables**')
                else:
                    st.info('**Category : Fruit**')
                st.success("**Predicted : " + result + '**')
                cal = fetch_calories(result)
                if cal:
                    st.warning('**' + cal + ' (100 grams)**')
        else:
            st.error("Error saving image. Please try again.")


# Run the app
run()
