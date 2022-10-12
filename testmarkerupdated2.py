from flask import Flask, request, jsonify

app = Flask(import_name=__name__)

print("LOG ---- Importing libraries....")

import requests
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import pandas as pd
import numpy as np
import os
#from sklearn.model_selection import train_test_split
# Import modules to evaluate the metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc
#from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
#from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses


print("LOG ---- libraries imported....")

#loading trained models
#WE CAN SAY FOR A PARTICULAR COURSE WE LOAD THAT MODEL, SO EACH COURSE WILL HAVE A PARTICULAR MODEL FOR EACH SCHOOL
#SO EACH COURSE HAS A COURSE CODE
#THAT COURSE CODE IS USED TO NAME THE MODEL FOR THAT COURSE
model = SentenceTransformer('hhhhhhh.model')
#GET Functions
def get_theory_id(data):
    print("LOG ---- loading Theory IDs....")
    ID = [data[i]["id"] for i in range(0,len(data))]
    return ID

def get_theory_answer(data):
    print("LOG ---- loading teacher's theory answers....")
    answer = [data[i]["answers"] for i in range(0,len(data))]
    return answer


def get_theory_solution(data):
    print("LOG ---- loading student's theory answers....")
    solution = [data[i]["solution"] for i in range(0,len(data))]
    return solution





#Process Functions
def process_score_theory(student, teacher):
    teachers_answers_list = teacher
    students_answers_list = student

    print("LOG ---- creating AI model ....")
    #tfidf_vectorizer = TfidfVectorizer()
    print("LOG ---- creating AI model [done] ....")

    similarity = []
    scores = []

    


    student_answer_matrix = trained_model.transform(students_answers_list)
    teacher_answer_matrix = []

    for i in range(0,len(teachers_answers_list)):
        ooo = model.encode(teachers_answers_list[i])
        teacher_answer_matrix.append(ooo)


    student_answer_matrix = model.encode(students_answers_list)


    teacher_answer_matrix2 = []
    teacher_answer_matrix2 = np.array(teacher_answer_matrix)
    print(teacher_answer_matrix2.shape)
    print(student_answer_matrix.shape)

    print("LOG ---- comparing student answers with teachers answers and computing scores ....")
    [similarity.append(cosine_similarity(student_answer_matrix, teacher_answer_matrix2[i])) for i in range(0,len(teacher_answer_matrix2))]
    print("LOG ---- comparing student answers with teachers answers and computing scores [done] ....")

    for i in range(0,len(similarity[0])):
        if max(similarity[0][i]) >= 0.8:
            mark = 2
        else:
            mark = 0
        scores.append(mark)
        print(max(similarity[0][i]))
    print(scores)
    return scores





#Post Function
'''this is the part that needs looking at '''
def post_score_theory(question, score):
    sum_score = sum(score)
    questions = []
    for a, b in zip(question, score):
        if b == 2:
            g=True
        else:
            g=False
        url = "https://smartcbt.herokuapp.com/tresult/"
        theory = {'id': a, 'isCorrect': g}
        questions.append(theory)
    return {"score":sum_score,"questions":questions}


@app.route("/score", methods=['GET', 'POST'])
def echo():
    if request.method == 'POST':
        data = request.get_json()
        print(data)

            #Call dGET functions
        Theory_id = get_theory_id(data)
        print(Theory_id)
        teachers_answers_theory = get_theory_answer(data)
        student_answers_theory = get_theory_solution(data)

        #Call PROCESS function
        process_theory = process_score_theory(student_answers_theory,teachers_answers_theory)

        #Call POST function
        post_theory = post_score_theory(Theory_id, process_theory)

        return jsonify(post_theory)
    else:
        return "THIS ENPOINT ACCEPTS POST REQUESTS ONLY"


if __name__ == "__main__":
    app.run(debug=True,port=5000)


    