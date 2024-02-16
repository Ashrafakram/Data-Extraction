#!/usr/bin/env python
# coding: utf-8

# In[29]:


from bs4 import BeautifulSoup
import pandas as pd
import requests
import xlsxwriter
import xlrd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from newspaper import Config,Article
import re
import enchant,sys
from nltk.tokenize import RegexpTokenizer , sent_tokenize
from nltk.corpus import stopwords


def textExtration():
    Excel_output = []
    sia = SentimentIntensityAnalyzer()
    filename = "C:\\Users\\Dell\\Downloads\\Input.xlsx"
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)    
    config = Config()
    while(True):
        for r in range(1, sheet.nrows):
            try:
                Url_ID = sheet.cell(r,0).value

                urls = sheet.cell(r, 1).value
                print(urls)
                article = Article(urls, config = config)
                article.download()
                article.parse()
                text = article.text
            except :
                pass
    #         print(article.title)
    
    #######sentiment analyasis using text blob########
            subjective,Polarity,negativescore,positivescore,neutralscore,compoundscore = PolarityScore(text)
        
    #######average length ##########
            avg_sentence_len = avg_sentence_length(text)
    #########syllable count#####################
            syllab_count = syllable_count(text) 
    ########################### fog_index#############
            fog_index,index_ = fogindex(text,syllab_count)
      
    ##########word count and sentence count###########################
            
            Word_count = len(text.split())
            print("wordcount:",Word_count)
            
            perctage_complex = perc_complex(text)
            print("perc_complex:",perctage_complex)
            
            personal_pronouns = cal_personal_pronouns(text)
            print("cal_personal_pronouns:",personal_pronouns)
            
            avg_word_length_ = avg_word_length(text)
            print("avg_word_length",avg_word_length_)
            
            av_sen_word_count_ = av_sen_word_count(text)
            print("av_sen_word_count_",av_sen_word_count_)
            
            text_file = f"{Url_ID}.txt"
            print("Text File:",text_file)
            with open(text_file,"w",encoding='utf-8') as file:
                file.write(text)
                
            dt_lst = [Url_ID,urls,positivescore,negativescore,Polarity,subjective,avg_sentence_len,perctage_complex,
                      fog_index,av_sen_word_count_,perctage_complex,Word_count,syllab_count,personal_pronouns,avg_word_length_]
            Excel_output.append(dt_lst)
    #         req = requests.get(urls)
    #         print(req)
    #         soup = BeautifulSoup(req.text,href=True)
    #         print(soup)
        break
    excel_data = pd.DataFrame(Excel_output,columns=["URL_ID","URL","POSITIVE SCORE","NEGATIVE SCORE","POLARITY SCORE","SUBJECTIVITY SCORE","AVG SENTENCE LENGTH","PERCENTAGE OF COMPLEX WORDS","FOG INDEX","AVG NUMBER OF WORDS PER SENTENCE","COMPLEX WORD COUNT","WORD COUNT","SYLLABLE PER WORD","PERSONAL PRONOUNS","AVG WORD LENGTH"])
    excel_path = r"C:\Users\Dell\Output_excel.xlsx"
    with pd.ExcelWriter(excel_path,engine='xlsxwriter') as writer:
        excel_data.to_excel(writer,sheet_name = "sheet_1" , index= False)
            
        
################################################################################################################################
def PolarityScore(text):
    """
    this block of code returns subiective , polarity , negative, neutral, positive and compond score of the text
    """
    try:
        sia = SentimentIntensityAnalyzer()
        blob = TextBlob(text)
        subjective = blob.sentiment.subjectivity
        Polarity = blob.sentiment.polarity
        print("POLARITY SCORE and SUBJECTIVITY SCORE:",subjective)
        print("POLARITY SCORE and SUBJECTIVITY SCORE:",Polarity)        
        senti = sia.polarity_scores(text) 
#         print(senti)
        negativescore = senti['neg']
        positivescore = senti['pos']
        neutralscore = senti['neu']
        compoundscore = senti['compound']
        print("Sentiment:",negativescore)
        
    except:
        pass
    return subjective,Polarity,negativescore,positivescore,neutralscore,compoundscore
################################################################################################################################
def avg_sentence_length(text):
    try:
        avg = len(text)
        print("Average Length:",avg)
        sentences = text.split(".") #split the text into a list of sentences.
        words = text.split(" ") #split the input text into a list of separate words
        if(sentences[len(sentences)-1]==""): #if the last value in sentences is an empty string
            average_sentence_length = len(words) / len(sentences)-1
        else:
            average_sentence_length = len(words) / len(sentences)
        print("Average Sentence Length:",average_sentence_length)
    except:
        pass
    
    if average_sentence_length == '':
        try:
            words = len(tokenz(text))
            sentences = len(text.split('.'))
            average_sentence_length = words / sentences
        except:
            pass
    return average_sentence_length

#################################################################################################################################

def syllable_count(text):
    try:
        syllablecount = 0
        beg_each_Sentence = re.findall(r"\.\s*(\w+)", text)
        capital_words = re.findall(r"\b[A-Z][a-z]+\b", text)
        words = text.split()
        for word in words:
            if word not in capital_words and len(word) >= 3: #all lower case words
#                         print(word)

                if syllables(word) >= 3 and len(split(word)) == 1:
                    syllablecount += 1

            if word in capital_words and word in beg_each_Sentence: #beginning of each sentence is uppercase

                if syllables(word) >= 3:
                    syllablecount += 1
        print("SyllableCount:",syllablecount)
    except:
        syllablecount = 0
        pass

    if syllablecount == 0:
        words = tokenz(text)
        words_cnt = len(words)
        vowels = 0
        for word in words:
            if word.endswith(('es','ed')):
                pass
            else:
                for l in word:
                    if (l == 'a' or l == 'e' or l == 'i' or l == 'o' or l == 'u'):
                        vowels += 1
        syllablecount = vowels / words_cnt
        print("syllablecount",syllablecount)
    return syllablecount
################################################################################################################################
def fogindex(text,syllablecount):
   
    word_count = len(re.findall("[a-zA-Z-]+", text))
    print("Word Count:",word_count)
#################################################      
    sentence_count = (len(re.split("[.!?]+", text))-1)
    print("Sentence Count:",sentence_count)
###############################################################        
    try:
        fog_index_calculated = ((word_count/sentence_count) + syllablecount)*0.4
        gunning_fog_index = ((word_count/sentence_count) + 100*(syllablecount/word_count))*0.4
    except ZeroDivisionError:
        fog_index_calculated = gunning_fog_index = 0
    print("Fog Calculation:",fog_index_calculated, gunning_fog_index)
        
    
    return fog_index_calculated,gunning_fog_index

################################################################################################################## ###############

def syllables(word):
    word = word.lower()
    word = word + " "  # word extended
    length = len(word)
    ending = ["ing ", "ed ", "es ", "ous ", "tion ", "nce ", "ness "]  # not included in complex words
    vowels = "aeiouy"

    for end in ending:
        x = word.find(end)
        if x > -1:
            x = length - x
            word = word[:-x]
    syllable_count = 0
    if word[-1] == " ":
        word = word[:-1]
    # removing the extra " " at the end if failed and dropping last letter if e
    if word[-1] == "e":
        try :
            if word[-3:] == "nce" and word[-3:] == "rce":
                syllable_count = 0

            elif word[-3] not in vowels and word[-2] not in vowels and word[-3:] != "nce" and word[-3:] != "rce":
                if word[-3] != "'":
                    syllable_count += 1  # e cannot be dropped as it contributes to a syllable
            word = word[:-1]
        except IndexError:
            syllable_count += 0

    one_syllable_beg = ["ya", "ae", "oe", "ea", "yo", "yu", "ye"]
    two_syllables = ["ao", "uo", "ia", "eo", "ea", "uu", "eous", "uou", "ii", "io", "ua", "ya", "yo", "yu", "ye"]
    last_letter = str()  # last letter is null for the first alphabet
    for index, alphabet in enumerate(word):
        if alphabet in vowels:
            current_combo = last_letter + alphabet
            if len(current_combo) == 1:  # if it's the first alphabet
#                 print(word)
                if word[1] not in vowels:  # followed by a consnant, then one syllable
                    syllable_count += 1
                    last_letter = word[1]
                else:
                    syllable_count += 1  # followed by a vowel
                    last_letter = alphabet

            else:
                if current_combo in two_syllables:
                    try:
                    # if they're only 1 syllable at the beginning of a word, don't increment
                        if current_combo == word[:2] and current_combo in one_syllable_beg:
                            syllable_count += 0
                        elif word[index - 2] + current_combo + word[index + 1] == "tion" or word[index - 2] + current_combo + \
                                word[index + 1] == "sion":  # here io is one syllable :
                            syllable_count += 0

                        else:
                            syllable_count += 1  # vowel combination forming 2 syllables

                        last_letter = alphabet
                    except IndexError:
                        syllable_count += 0

                else:  # two vowels as well as non vowel combination
                    if last_letter not in vowels:
                        syllable_count += 1
                        last_letter = alphabet

                    else:
                        last_letter = alphabet


        else:
            last_letter = alphabet

    if word[-3:] == "ier":  # word ending with ier has 2 syllables
        syllable_count += 1

    return syllable_count

def split(compound_word, language='en_US'):

    words = compound_word.split('-')

    word = ""

    for x in words:
        word += x

    result = __split(word, language)

    if result == compound_word:
            return [result]

    return result
def __concat(object1, object2):

    if isinstance(object1, str) or isinstance(object1, unicode):
        object1 = [object1]
    if isinstance(object2, str) or isinstance(object2, unicode):
        object2 = [object2]
    return object1 + object2


def __capitalize_first_char(word):

    return word[0].upper() + word[1:]


def __split(word, language='en_US'):

    dictionary = enchant.Dict(language)
    max_index = len(word)

    if max_index < 3:
        return word

    for index, char in enumerate(word, 2):

        left_word = word[0:index]
        right_word = word[index:]

        if index == max_index - 1:
            break

        if dictionary.check(left_word) and dictionary.check(right_word):
            return [compound for compound in __concat(left_word, right_word)]

    return word

def tokenz(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = str(text)
    stopword = stopwords.words()
    tokens = tokenizer.tokenize(text)
    filter = []
    for token in tokens:
        if token not in stopword:
            filter.append(token)
    return filter

def perc_complex(text):
#     print(text)
    tokens = tokenz(text)
#     print(tokens)
    complex_cnt = 0
    for token in tokens:
        vowels = 0
        if token.endswith(('es','ed')):
            pass
        else:
            for l in token:
                if (l == 'a' or l == 'e' or l == 'i' or l == 'o' or l == 'u'):
                    vowels += 1
            if vowels > 2:
                complex_cnt += 1
#         print(token)
#         print(complex_cnt)
    if len(token) != 0:
        return complex_cnt/len(token)
    
def cal_personal_pronouns(text):
    words = tokenz(text)
    pronoun_re = r'\b(I|my|we|us|ours)\b'
    matches = re.findall(pronoun_re,text)
    return len(matches)

def avg_word_length(text):
    words = tokenz(text)
    charcnt = 0
    for word in words:
        charcnt += len(word.strip())
    if len(words) != 0:
        return charcnt / len(words)

def av_sen_word_count(text):
    try:
        sentence = re.split("[.!?]+", text)
        avg_sentence = len(sentence)
        avg_word = 0
        for sentence in sentence:
            words = tokenz(sentence)
            avg_word += len(words)
        return avg_word / avg_sentence
    except:
        pass
        
    
            
if __name__ == '__main__':
    textExtration()






