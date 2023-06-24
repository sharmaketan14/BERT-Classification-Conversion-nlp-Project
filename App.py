import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request


def classify_sentence(sentence, model):
    sentence = np.array([[sentence]])
    print(sentence)
    y_predict = model.predict(sentence)
    y_predict = y_predict.flatten()
    y_predict = np.where(y_predict > 0.5, 1, 0)

    return y_predict[0]

def prepare_data():
  import pandas as pd
  final_dataset = pd.read_csv("/content/drive/MyDrive/Project/Processed_Data/fin.csv")
  X = final_dataset["Comments"]
  y = final_dataset["Type"]
  final_dataset.head()

  final_dataset.groupby('Type').describe()
  final_dataset["Type"].value_counts()

    #df_roast = final_dataset[final_dataset["Type"] == 0]
  df_toast = final_dataset[final_dataset["Type"] == 1]
    #df_roast_downsampled = df_roast.sample(df_toast.shape[0])

  sentences = []
  for i in df_toast["Comments"] :
    sentences.append(i)

  return sentences

def sentence_vec(sentences):
  model_name = 'bert-base-nli-mean-tokens'
  from sentence_transformers import SentenceTransformer

  md = SentenceTransformer(model_name)

  Sentence_vecs = md.encode(sentences)

  return Sentence_vecs

def get_convert_sentence(str1, Sentence_vecs, sentences):
  model_name = 'bert-base-nli-mean-tokens'
  from sentence_transformers import SentenceTransformer

  md = SentenceTransformer(model_name)

  sen = md.encode(str1)

  from sklearn.metrics.pairwise import cosine_similarity
  result = cosine_similarity([sen], Sentence_vecs[:])

  m = 0
  final_conversion = ""
  for i in range(0, len(result[0])):
    if(m<result[0][i]):
      m = result[0][i]
      final_conversion = sentences[i]

  return final_conversion


def final(str1):
    from keras.models import load_model
    #sentence = input("Enter the sentence: ")
    model = load_model("/content/drive/MyDrive/Project/model")
    result = classify_sentence(str1, model)
    
    if(result == 1):
        print("Roast Sentence")
        sentences = prepare_data()
        model_name = 'bert-base-nli-mean-tokens'
        from sentence_transformers import SentenceTransformer

        md = SentenceTransformer(model_name)

        vec_encodings = md.encode(sentences)
        final_convert = get_convert_sentence(sentence, vec_encodings, sentences)
        return final_convert
    else:
        return "Already Toast :)"


from flask import Flask, jsonify, request

app = Flask(__name__)


return_file = [
    { 'sentence': "Whats UP!"}
]


@app.route('/', methods=['GET'])
def get_incomes():
  
  return "<h1>Hello World</h1>"


# @app.route('/convert', methods=['POST'])
# def get_sentence():
#     str1 = request.form['sentence']
#     return '', 204

if __name__ == '__name__':
  app.run(port=5000)

app.run(port="5000", debug=True)