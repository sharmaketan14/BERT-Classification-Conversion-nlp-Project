def classify_sentence(sentence, model):
    sentence = np.array([[sentence]])
    print(sentence)
    #y_predict = model.predict(sentence)
    y_predict = y_predict.flatten()
    y_predict = np.where(y_predict > 0.5, 1, 0)
    
    return y_predict[0][0]