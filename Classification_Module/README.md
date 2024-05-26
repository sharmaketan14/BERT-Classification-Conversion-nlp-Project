This project is based on IIIT Hyderabad Reasearch Paper "Tackling Targeted Negative Speech on Social Media through Text Classification and Style Transfe" by "Ravisimar Sodhi".
We use BERT (Bidirectional Encoder Representations from Transformers) model to classify the roast and toast sentences and then using BERT model vector encodings we use cosine similarity to find the most suitable toast sentence that can be replaced for that toast sentence.
The Project is complete but the model can be optimized more by using RoBERTA (A Robustly Optimized BERT Pretraining Model) and using more Social Media comment-reply dataset.
The Overview is that the Model takes a comment and classifies it as toast or roast, if classified as roast it replaces or converts the comment to toast.
For the API development, Flask has been used and integrated with NGROK as the development wqs being done on Google Colab so, making the server's home address public was mandatory.

Technologies Used: Tensorflow, BERT Model, Neural Networks, Flask API

Created By : Ketan Sharma
Inspired and Credits for Guidance : "Tackling Targeted Negative Speech on Social Media through Text Classification and Style Transfe" by "Ravisimar Sodhi".
