from source import pdftotext, parser
# from flask import Flask, request
import os


# app = Flask(__name__)


def load_resume():
    resume = []
    for file in os.listdir('data'):
        path = os.path.join('data', file)
        document = pdftotext.convert_pdf_to_txt(path)
        resume.append([file.replace('.pdf', ''), document])
    return resume




if __name__ == "__main__":
    resume = load_resume()
    for r in resume:
        print("Resume Name:", r[0])
        print("Emaial id:", parser.getEmail(r[1]))
        print("Phone Number:", parser.getPhone(r[1]))
        tokens, lines, sentences = parser.preprocess(r[1])
        # print("Lines:", lines)
        print("Person Name:", parser.getName(lines))
        # exit()
        print("X"*20)