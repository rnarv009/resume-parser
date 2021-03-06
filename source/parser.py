import re
# from pdftotext import convert_pdf_to_txt
import nltk


def preprocess(document):
    lines = [l.strip() for l in document.split("\n") if len(l) > 0]
    lines = [nltk.word_tokenize(l) for l in lines]
    lines = [nltk.pos_tag(l) for l in lines]
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tokens = sentences
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    dummy = []
    for l in tokens:
        dummy += l
    tokens = dummy
    return tokens, lines, sentences


def getPhone(document):
    pattern = re.compile(
        r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')

    match = pattern.findall(document)
    # match = [re.sub(r'[,.]', '', el) for el in match if len(re.sub(r'[()\-.,\s+]', '', el)) > 6]
    # match = [re.sub(r'\D$', '', el).strip() for el in match]
    match = [num.strip() for num in match if len(re.sub(r'\D', '', num)) <= 15]
    number = [num.strip() for num in match if len(num.replace('-', '')) >= 10]

    return number[0]


def getEmail(document):
    """
    regex for finding the email from the documents(string)
    :param document:
    :return list of emails:
    """
    email = re.findall(r'\S+@\S+', document)
    return email[0]


if __name__ == "__main__":
    # document = convert_pdf_to_txt('../data/Lakshman patel 1-converted.pdf')
    # document = convert_pdf_to_txt('../data/Shashi_Patel.pdf')
    document = convert_pdf_to_txt('../data/Rahul_Verma.pdf')
    tokens, lines, sentences = preprocess(document)
    # print(sentences[0])
    print("Emaial id:", getEmail(document))
    print("Phone Number:", getPhone(document))
