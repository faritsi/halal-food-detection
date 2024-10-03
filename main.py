import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
import joblib
# import easyocr
from PIL import Image
import os
import pandas as pd
import json
import time
import altair as alt
import cv2
import sqlite3
import uuid
# import paddle

# from sklearn.metrics import accuracy_score, classification_report
import os

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Load data
@st.cache_data

def load_model2():
    from tensorflow.keras.models import load_model
    return load_model("D:\Model\daset\jner\model-baru")
# def perform_ocr(image_path):
#     results = ocr.ocr(image_path)
#     return results

# Load model XGboost
def load_model():
    loaded_model = joblib.load('D:/Model/daset/xgb2/xgboost_food_model.joblib')
    loaded_vectorizer = joblib.load('D:/Model/daset/xgb2/vectorizer.joblib')
    loaded_label_encoder = joblib.load('D:/Model/daset/xgb2/label_encoder.joblib')
    return loaded_model, loaded_vectorizer, loaded_label_encoder
# Function to predict the names of the labels for a single food item string using the loaded model
def predict_food_classification_loaded_single(food_description, model, vectorizer, label_encoder):
    input_vector = vectorizer.transform([food_description])
    predicted_label_index = model.predict(input_vector)[0]
    predicted_label_name = label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_label_name
def predict_food_classification_loaded_multiple(food_descriptions, model, vectorizer, label_encoder):
    descriptions_list = [description.strip() for description in food_descriptions.split(',')]
    input_vectors = vectorizer.transform(descriptions_list)
    predicted_label_indices = model.predict(input_vectors)
    predicted_label_names = label_encoder.inverse_transform(predicted_label_indices)
    return predicted_label_names
@st.cache_resource
def load_data():
    return load("D:\Model\data.npy", allow_pickle=True)

# Load data structures and models
PaddleOCR = 'D:\Model\PaddleOCR'
from paddleocr import PaddleOCR, draw_ocr
model_ocr = PaddleOCR(use_angle_cls=True, lang='en')
out_path = "D:\Model\output"
# out_path = 'D:/Model/out'
font = 'D:/Model/PaddleOCR/doc/fonts/simfang.ttf'
token2idx = np.load('D:\Model\daset\jner\model-baru\data\wordidx.npy', allow_pickle=True).item()
input_length = int(np.load('D:\Model\daset\jner\model-baru\data\maxlen.npy'))
tag2idx = np.load('D:\Model\daset\jner\model-baru\data\etagidx.npy', allow_pickle=True).item()
idx2token = np.load('D:\Model\daset\jner\model-baru\data\idx2token.npy', allow_pickle=True).item()
idx2tag = np.load('D:\Model\daset\jner\model-baru\data\idx2tag.npy', allow_pickle=True).item()
vectorizer = CountVectorizer()
# loaded_model1 = tf.keras.models.load_model('/content/drive/MyDrive/model1.h5')
# loaded_model2 = tf.keras.models.load_model('/content/drive/MyDrive/model2.h5')
def preprocess_image(img_path):
    # Read image
    img = cv2.imread(img_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow(thresh)
    return thresh
  
def save_ocr(img_path, result, font, output, save='false', out_path=""):
    # save_path = os.path.join(out_path, img_path.split('/')[-1].split('.')[0] + '-output.jpg')

    # image = cv2.imread(img_path)
    # gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Extracting boxes, texts, and their scores from the output list.
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # Plotting the outputs using PaddleOCR in-built function.
    im_show = draw_ocr(cv2.imread(img_path), boxes, txts, scores, font_path=font)
    # Save Output Deteciotns
    if save:
        img = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        file_name = out_path + 'output-' + str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(os.path.join(output, file_name), img)
    # Saving the output image.
    # cv2.imwrite(save_path, im_show)
    # Menggabungkan elemen-elemen dalam list txts menjadi satu string
    output_string = ' '.join(txts)
    return output_string, im_show
    # Menampilkan output string
    # print(output_string)
    # # Displaying the output.
    # img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

# Function to perform NER on user input
def perform_ner(input_text, model):
    # input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)
    # input_text = input_text.lower()

    encoded = [token2idx.get(t, 0) for t in input_text.split()] + [0] * input_length
    encoded = encoded[:input_length]
    sample = np.array(encoded).reshape((1, input_length))
    prediction = model.predict(sample)
    ner = np.squeeze(np.argmax(prediction, axis=-1)[:len(input_text.split())])
    # ner = np.argmax(prediction, axis=-1)[0][:len(input_text.split())]

    # Group consecutive "B-ingredients" and "I-ingredients" tags into a single "ingredients" tag
    result = []
    current_tag = ""
    current_entity = ""
    
    for token, tag_idx in zip(input_text.split(), ner):
        tag = idx2tag[tag_idx]
        
        if tag.startswith("B-"):
            # Start of a new entity
            if current_entity:
                result.append((current_entity, "ingredients"))
            current_entity = token
            current_tag = tag.replace("B-", "")
            print(f"tag awal: {current_entity},{current_tag}")
        elif tag.startswith("I-"):
            # Continuation of the current entity
            if current_tag == tag.replace("I-", ""):
                current_entity += " " + token
                print(f"tag kedua: {current_entity},{current_tag}")
            else:
                result.append((current_entity, "ingredients"))
                current_entity = token
                current_tag = tag.replace("I-", "")
                print(f"tag ketiga: {current_entity},{current_tag}")

        else:
            # Non-entity token
            if current_entity:
                result.append((current_entity, "ingredients"))
                current_entity = ""
    
    # Add the last entity if present
    if current_entity:
        result.append((current_entity, "ingredients"))

    return result

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create a table
def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS records
                          (id INTEGER PRIMARY KEY, im_show BLOB, ner_result TEXT, status TEXT, ocr_comparison TEXT, ner_comparison TEXT)''')
    except sqlite3.Error as e:
        print(e)

def convertToBinaryData(filename): 
      
    # Convert binary format to images  
    # or files data 
    with open('filename', 'r', encoding='latin1') as file: 
        blobData = file.read() 
    return blobData 

def get_last_image(folder_path):
    # Dapatkan daftar file dalam folder
    files = os.listdir(folder_path)
    # Filter hanya file gambar
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Urutkan file berdasarkan waktu modifikasi
    sorted_images = sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    # Dapatkan nama file gambar terakhir
    last_image = sorted_images[0] if sorted_images else None
    return last_image

# Function to insert a record
def insert_record(conn, im_show, ner_result, status, ocr_comparison, ner_comparison):
    try:
        cursor = conn.cursor()
        sql_insert = """ INSERT INTO records (im_show, ner_result, status, ocr_comparison, ner_comparison) VALUES (?, ?, ?, ?, ?)"""
        # cursor.execute('''INSERT INTO records (im_show, ner_result, status) VALUES (?, ?, ?)''', (im_show, ner_result, status))
        last_image = get_last_image(out_path)
        if last_image:
            image_path = os.path.join(out_path, last_image)
            # Baca gambar terakhir dan simpan datanya
            with open(image_path, 'rb') as file:
                ocr_simpan = file.read()
            data_tuple = (ocr_simpan, ner_result, status, ocr_comparison, ner_comparison)
            cursor.execute(sql_insert, data_tuple)
            conn.commit()
            print("Image and file inserted successfully as a BLOB into a table")
            return cursor.lastrowid
        else:
            print("No image found in the folder.")
            return None
    except sqlite3.Error as e:
        print(e)

# Function to retrieve all records
def retrieve_records(conn, offset, limit):
    try:
        cursor = conn.cursor()
        cursor.execute('''SELECT * FROM records LIMIT ? OFFSET ?''', (limit, offset))
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(e)

# Function to save check result to CSV
def save_check_result(model_ner, ner_result, status):
    data = {'Model': [model_ner], 'NER Result': [ner_result], 'Status': [status]}
    df = pd.DataFrame(data)

    if not os.path.isfile('D:\Model\save_history\check_results.csv'):
        df.to_csv('D:\Model\save_history\check_results.csv', index=False)
    else:
        df.to_csv('D:\Model\save_history\check_results.csv', mode='a', header=False, index=False)

# Function to show all checks
# def show_all_checks():
#     if os.path.isfile('D:\Model\save_history\check_results.csv'):
#         st.title("Checks History")
#         df = pd.read_csv('D:\Model\save_history\check_results.csv')
#         st.dataframe(df)
#     else:
#         st.warning("No checks recorded yet.")

def colorize_label(label):
    if label.lower() == 'haram':
        return f'<span style="color: white; padding: 5px 10px; margin: 15px auto; background-color: red; border: 1px solid white; border-radius: 10px;">{label}</span>'
    elif label.lower() == 'syubhat':
        return f'<span style="color: white; padding: 5px 10px; margin: 15px auto; background-color: orange; border: 1px solid white; border-radius: 10px;">{label}</span>'
    else:
        return f'<span style="color: white; padding: 5px 10px; margin: 15px auto; background-color: green; border: 1px solid white; border-radius: 10px;">{label}</span>'

def cekduls(input_text, model):
    encoded = [token2idx.get(t, 0) for t in input_text.split()] + [0] * input_length
    print("OOV:", [t for t in input_text.split() if t not in token2idx])
    encoded = encoded[:input_length]

    sample = np.array(encoded).reshape((1, input_length))
    prediction = model.predict(sample)
    ner = np.squeeze(np.argmax(prediction, axis=-1)[:len(input_text.split())])
    print([(t, idx2tag[n]) for t, n in zip(input_text.split(), ner)])

def normalize_text(text):
    # Ubah semua huruf menjadi huruf kecil dan hapus karakter khusus
    return re.sub(r'\W+', ' ', text.lower()).strip()

# Define CSS for custom styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the custom CSS
local_css("style.css")

def calculate_word_proportion(output_string, res_pre, ner_result):   
    output_words = set(normalize_text(output_string).split())
    res_pre_words = set(normalize_text(res_pre).split())
    ner_result_words = set(normalize_text(ner_result).split())

    print(output_string)
    print(res_pre_words)
    print(ner_result_words)
    
    # Cari kata yang cocok
    matching_words = output_words & res_pre_words & ner_result_words
    total_matching_words = len(matching_words)
    
    # Hitung proporsi
    ocr_comparison = f"{len(res_pre_words)}/{len(output_string.split())}" if res_pre_words else "0/0"
    ner_comparison = f"{len(ner_result_words)}/{len(output_string.split())}" if ner_result_words else "0/0"
    
    return ocr_comparison, ner_comparison

    # return {
    #     "ocr_comparison": ocr_comparison,
    #     "ner_comparison": ner_comparison,
    #     "total_matching_words": total_matching_words,
    #     "matching_words": matching_words
    # }


# Streamlit app with dynamic tabs
def main():
    conn = create_connection("halalCheck4.db")
    if conn is not None:
        # Create table
        create_table(conn)
        st.title("Halal Food Checker :white_check_mark:")

        loaded_model, loaded_vectorizer, loaded_label_encoder = load_model()
        model_ner = load_model2()
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("")

        # selected_model = st.selectbox("Choose Model", ["Model 1", "Model 2"])

        # if selected_model == "Model 1":
        #     model_ner = load_model1()
        # else:
        #     model_ner = load_model2()
        # Dynamic Tabs
        tab1, tab2, tab3 = st.tabs(["Check Status", "History", "Model Result"])
        with tab1:
            if st.button("Check Status"):
                if uploaded_file is None:
                    st.warning("Please upload an image.")
                else:
                    image = Image.open(uploaded_file)
                    image_path = f"uploaded_image.{uploaded_file.type.split('/')[1]}"
                    image.save(image_path)
                    res = model_ocr.ocr(preprocess_image(image_path))
                    output_string, im_show = save_ocr(image_path, res, font, out_path)
                    # output_string, im_show = save_ocr()
                    print(output_string)
                    st.header("OCR Results:")
                    # st.write(f"{output_string}")
                    # combined_results = " ".join([result[0] for result in output_string])
                    # output_string_cleaned = combined_results.replace(" ", ",")
                    pre_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', output_string)
                    st.image(im_show, caption='Enter any caption here')
                    
                    col1, col2 = st.columns(2)
                    
                    res_pre = pre_text.lower()
                    col1.header("OCR Results:")
                    col1.write(f"{res_pre}")
                    cekdss = cekduls(res_pre, model_ner)
        
                    result_model2 = perform_ner(res_pre, model_ner)

                    ingredients_list = [entity for entity, tag in result_model2 if tag == "ingredients"]
                    ner_result = ", ".join(f"{ingredient}" for ingredient in ingredients_list)
                    col2.header("NER Result:")
                    col2.write(f"{ner_result}")
                    ocr_comparison, ner_comparison = calculate_word_proportion(output_string, res_pre, ner_result)

                    if output_string:
                        predicted_labels = predict_food_classification_loaded_multiple(
                            ner_result, loaded_model, loaded_vectorizer, loaded_label_encoder
                        )

                        is_haram = any(label.lower() == 'haram' for label in predicted_labels)
                        # has_syubhat = any(label.lower() == 'syubhat' for label in predicted_labels)

                        result_tags = ", ".join([f'{description} {colorize_label(label)}' for description, label in zip(ner_result.split(','), predicted_labels)])

                        if is_haram:
                            # st.text_area(label='Ingredients of foods', value=result_tags, disabled=True, unsafe_allow_html=True)
                            st.header("Status")
                            st.markdown(f"<div style='padding: 10px auto;'>{result_tags}</div>", unsafe_allow_html=True)
                            st.header("Result:")
                            st.error("The food is Haram because it contains haram ingredients.")
                            insert_record(conn, im_show, ner_result, "Haram", ocr_comparison, ner_comparison)
                            # save_check_result(model_ner, ner_result, "Haram")
                        # elif has_syubhat:
                        #     # st.text_area(label='Ingredients of foods', value=result_tags, disabled=True, unsafe_allow_html=True)
                        #     st.header("Status")
                        #     st.markdown(f"<div style='padding: 10px auto;'>{result_tags}</div>", unsafe_allow_html=True)
                        #     st.warning("Prediction result: The food is Confuse due to uncertain/ambiguous ingredients.")
                        #     # save_check_result(model_ner, ner_result, "Syubhat")
                        else:
                            # st.text_area(label='Ingredients of foods', value=result_tags, disabled=True, unsafe_allow_html=True)
                            st.header("Status")
                            st.markdown(f"<div style='padding: 10px auto;'>{result_tags}</div>", unsafe_allow_html=True)
                            st.header("Result:")
                            st.success("The foods are halal. Safe for consumption")
                            # save_check_result(model_ner, ner_result, )
                            # status = "Halal"
                            insert_record(conn, im_show, ner_result, "Halal", ocr_comparison, ner_comparison)
                    else:
                        st.warning("Please enter at least one food description.")
                    
                    # res_pre_list = res_pre.split()
                    # ner_result_list = [word.strip() for word in ner_result.split(',')]
                    # st.header("Perbandingan:")
                    # st.write(f"OCR Comparison: {ocr_comparison}")
                    # st.write(f"NER Comparison: {ner_comparison}")
        with tab2:
            records_per_page = 5
            offset = st.session_state.get('offset', 0)

            prev_button, next_button = st.columns([1, 1])
            with prev_button:
                if st.button("Previous") and offset > 0:
                    offset -= records_per_page
                    st.session_state.offset = offset
            with next_button:
                if st.button("Next"):
                    offset += records_per_page
                    st.session_state.offset = offset

            records = retrieve_records(conn, offset, records_per_page)
            if not records:
                st.write("No records found.")
            else:
                for record in records:
                    record_id, im_show, ner_result, status, ocr_comparison, ner_comparison = record
                    st.header(f"No: {record_id}")
                    st.header(f"Image:")
                    st.image(im_show, caption='OCR Image')
                    st.header(f"Result:")
                    st.write(f"{ner_result}")
                    st.header(f"Status:")
                    st.write(f"<div style='background-color: blue; border-radius: 5px; padding: 0 auto; margin: 0 auto;'><p style='padding: 10px; margin: 10px; color: white; text-align: center'>{status}</p></div>", unsafe_allow_html=True)
                    # f"<div style='background-color: orange; border-radius: 5px; padding: 0 auto; margin: 0 auto;'><p style='padding: 10px; margin: 10px; color: white; text-align: center'>{record[3]}</p></div>", unsafe_allow_html=True
                    st.write(f"OCR Comparison: {ocr_comparison}")
                    st.write(f"NER Comparison: {ner_comparison}")
                    st.write("---")
        
        with tab3:
            ner = Image.open('D:/Model/model_res/NER.png')
            xg = Image.open('D:/Model/model_res/XG.png')
            xgt = Image.open('D:/Model/model_res/XGT.png')
            gr1 = Image.open('D:/Model/model_res/grapik1.png')
            gr2 = Image.open('D:/Model/model_res/grapik2.png')
            gr3 = Image.open('D:/Model/model_res/grapik3.png')
            gr4 = Image.open('D:/Model/model_res/grapik4.png')

            category = st.radio("Choose: ", ["NER", "XGBoost"])
            col1, col2 = st.columns(2)
            if category == "NER":
                st.header("Classification Report")
                st.image(ner)
                with col1:
                    st.header("Accuracy Chart")
                    st.image(gr1, caption='Enter any caption here')
                with col2:
                    st.header("Loss Chart ")
                    st.image(gr2, caption='Enter any caption here')
            elif category == "XGBoost":
                st.header("Classification Report")
                st.image(xg)
                with col1:
                    st.header("Accuracy Chart")
                    st.image(gr3, caption='Enter any caption here')
                with col2:
                    st.header("Loss Chart ")
                    st.image(gr4, caption='Enter any caption here')
    conn.close()
        

    # if selected_tab == "Check Status":
        
    # elif selected_tab == "Show All Checks":
    #     show_all_checks()

if __name__ == "__main__":
    main()

def format_ner_results(result):
    ingredients_list = [entity for entity, tag in result if tag == "ingredients"]
    formatted_result = ", ".join(f"{ingredient}" for ingredient in ingredients_list)
    return formatted_result

# if __name__ == "__main__":
#     main()
