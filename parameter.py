class Parameter:
    def __init__(self):
        self.data_cnf = "/Users/mm/Documents/Course_Information/Data_Mining/CorNet-master/configure/datasets/EUR-Lex.yaml"
        self.model_cnf = "/Users/mm/Documents/Course_Information/Data_Mining/CorNet-master/configure/models/XMLCNN-EUR-Lex.yaml"
        EURLex = "/Users/mm/Documents/Course_Information/Data_Mining/EUR-Lex/"
        self.models = EURLex + "models"
        self.emb = EURLex + "emb.npy"
        self.vocab = EURLex + "vocab.npy"
        self.train_texts = EURLex + "train_texts.npy"
        self.train_texts_txt = EURLex + "train_texts.txt"
        self.train_labels = EURLex + "train_labels.npy"
        self.train_labels_txt = EURLex + "train_labels.txt"
        self.token = EURLex + "token.txt"
        self.labels_binarizer = EURLex + "labels_binarizer"
        glove = "/Users/mm/Documents/Course_Information/Data_Mining/glove/"
        self.w2v_model = glove + "glove.840B.300d.gensim"



