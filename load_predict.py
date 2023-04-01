# In here we have done screening for RESUMES and COVER LETTER

import spacy
import joblib

from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
import re

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    text = ' '.join(text.split()) 

    return text.strip()


def cleanText(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text

def model_1(text, job_position, label_mapping):
    flag = False

    # loading saved pkl model file(First Part of our model Model)
    tfidf = joblib.load("ResumeFraserModelEncoding.pkl")
    model = joblib.load("ResumePhrasingModel.pkl")

    # cleaning our text
    text = cleanText(text)

    # predicting
    prediction = model.predict(tfidf.transform([text]))
    if label_mapping[float(prediction)] == job_position:
        print(f'Candidate is suitable for the {job_position} Job Position. Consider him for further interview rounds!')
        flag = True
    else:
        print('Candidate is not suitable')

    return flag

def model_2(text):
    # you can either use 'bert_based_uncased' or 'roberta_base'
    nlp = spacy.load('bert_based_uncased') # we passed in here the model we trained
    doc = nlp(text)
    for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}- {ent.text}')

def main():
    # getting text from our Resume or Cover Letter
    resume = pdf_reader('resume.pdf') # pass path to your Resume/Cover-Letter

    # mapping of all the Job Position for our model 1
    label_mapping = {0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
                     4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 7: 'Database',
                     8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer', 11: 'Electrical Engineering',
                     12: 'HR', 13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 
                     16: 'Mechanical Engineer', 17: 'Network Security Engineer', 18: 'Operations Manager', 19: 'PMO',
                     20: 'Python Developer', 21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'}
    
    # choose the Job Position you want to screen the Resume for, from the options in the mapping
    job_position = label_mapping[6]
    flag = model_1(resume, job_position, label_mapping)

    # if the resume is selected, let's get the information about the candidate
    if flag:
        model_2(resume)


if __name__ == '__main__':
    main()    