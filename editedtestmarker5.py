from flask import Flask, request, jsonify

app = Flask(import_name=__name__)

print("LOG ---- Importing libraries....")

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

print("LOG ---- libraries imported....")


#GET Functions
def get_theory_id(data):
    print("LOG ---- loading Theory IDs....")
    ID = [data["questions"][i]["id"] for i in range(0,len(data["questions"]))]
    return ID

def get_theory_answer(data):
    print("LOG ---- loading teacher's theory answers....")
    answer = [data["questions"][i]["answers"] for i in range(0,len(data["questions"]))]
    return answer


def get_theory_solution(data):
    print("LOG ---- loading student's theory answers....")
    solution = [data["questions"][i]["solution"] for i in range(0,len(data["questions"]))]
    return solution





#Process Functions
def process_score_theory(student, teacher):
    teachers_answers_list = teacher
    students_answers_list = student

    print("LOG ---- creating AI model ....")
    tfidf_vectorizer = TfidfVectorizer()
    print("LOG ---- creating AI model [done] ....")

    similarity = []
    scores = []

    

    print("LOG ---- training AI model ....")


    teachers_answers_list_trained = []
    for i in range(0,len(teachers_answers_list)):
        for b in range(0,len(teachers_answers_list[i])):
            teachers_answers_list_trained.append(teachers_answers_list[i][b])



    trained_model = tfidf_vectorizer.fit(teachers_answers_list_trained)
    #trained_model = trained_model.fit(teachers_answers_list)
    print("LOG ---- training AI model [done] ....")


    student_answer_matrix = trained_model.transform(students_answers_list)
    teacher_answer_matrix = []

    for i in range(0,len(teachers_answers_list)):
        ooo = trained_model.transform(teachers_answers_list[i])
        teacher_answer_matrix.append(ooo)

    #teacher_answer_matrix = teacher_answer_matrix.toarray()
    #student_answer_matrix = student_answer_matrix.toarray()
    teacher_answer_matrix2 = []
    """student_answer_matrix = np.array(student_answer_matrix)
    teacher_answer_matrix = np.array(teacher_answer_matrix)"""
    student_answer_matrix = student_answer_matrix.todense()
    for i in range(0,len(teacher_answer_matrix)):
        ooo = np.array(teacher_answer_matrix[i].toarray().sum(axis=0))
        teacher_answer_matrix2.append(ooo)


    print("ddddd ",np.array(teacher_answer_matrix2))
    teacher_answer_matrix2 = np.array(teacher_answer_matrix2)
    print("LOG ---- comparing student answers with teachers answers and computing scores ....")
    similarity.append(cosine_similarity(student_answer_matrix, teacher_answer_matrix2))
    print("LOG ---- comparing student answers with teachers answers and computing scores [done] ....")

    for i in range(0,len(similarity[0])):
        if max(similarity[0][i]) >= 0.7:
            mark = 2
        else:
            mark = 0
        scores.append(mark)
    score = scores
    print(score)
    return score





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

            #Call GET functions
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