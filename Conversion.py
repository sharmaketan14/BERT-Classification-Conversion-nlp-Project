

def prepare_data():
  import pandas as pd
  final_dataset = pd.read_csv("./Processed_Data/fin.csv")
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