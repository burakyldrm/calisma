# -*- coding: utf-8 -*-

from polyglot.text import Text
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import datetime
import time

def get_named_entities(text, lang="eng"):
	
	aylar = ['', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

	sents_for_hizmet = sent_tokenize(text)
	h_num = []
	hizmet_nums = []

	compiled_re_for_date = re.compile(r'(\d{2})[/.-](\d{2})[/.-](\d{4})$')
	compiled_re_for_clock = re.compile(r'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$')

	dates = []
	dates2 = []
	date_manuel = ""

	clocks = []
	clocks2 = []

	ctr = 0		
	for sent in sents_for_hizmet:
		words = word_tokenize(sent)
		for w in words:
			if test_date(w, compiled_re_for_date) == 1:
				dates.append(w)
			word = w.lower()

			if word in aylar:
				find_dates(word, words, ctr, dates)

			test_clock(w, compiled_re_for_clock, clocks)

			if test_hizmet_numarasi(w)==0:
				h_num = []
			else:
				hizmet_nums.append(w)

			ctr = ctr + 1
		ctr = 0
	'''
	for nums in hizmet_nums:	
		h_num.extend([("".join("Service Num: "), nums)])

	for date in dates:
		dates2.extend([("".join("Date: "), date)])

	for clock in clocks:
		clocks2.extend([("".join("Hour: "), clock)])
	'''
	ptext = Text(text, hint_language_code=lang)
	sentences = ptext.sentences
	entities = []
	for sentence in sentences:
				lang_sentence = Text(str(sentence), hint_language_code=lang)
				s_entities = [(" ".join(entity), entity.tag) for entity in lang_sentence.entities]
				entities.extend(s_entities)
	
	entities=list(set(entities))
	
	return entities + h_num + dates2 + clocks2

def test_hizmet_numarasi(word):
	if representsInt(word) == 1:
		if len(str(word)) > 7 and int(word) > 0:
			return 1
	return 0

def representsInt(s):
	try:
		int(s)
		return 1
	except ValueError:
		return 0

def test_date(datestring, pattern_date):
        try:
                mat=re.match(pattern_date, datestring)
                if mat is not None:
                        datetime.datetime(*(map(int, mat.groups()[-1::-1])))
                        return 1
        except ValueError:
                pass
        return 0

def test_clock(str, pattern_clock, clocks):
	found_bool = re.match(pattern_clock, str)
	if found_bool is None:
		pass
	else:
		clocks.append(str)

def find_dates(word, words, ctr, dates):
	day = " "
	month = " "
	year = " "
	if ctr+1 != len(words):
		if representsInt(words[ctr+1]) == 1 and len(words[ctr+1]) == 4:
			year = str(words[ctr+1])

		if representsInt(words[ctr-1]) == 1 and (len(words[ctr-1]) == 2 or len(words[ctr-1]) == 1):
			day = str(words[ctr-1])
	month = word.title()
	date_manuel = str(day) + " " + str(month) + " " + str(year)
	dates.append(date_manuel)

def format_response(resp_ner):
	locs=[]
	orgs=[]
	pers=[]
	resp={"orgs":orgs,
		"locs":locs,
		"pers":pers}
	
	for item in resp_ner:
		if item[1]=='I-ORG':
			orgs.append(item[0])
		elif item[1]=='I-LOC':
			locs.append(item[0])
		elif item[1]=='I-PER':
			pers.append(item[0])
	
	return resp

def getNamedEntityEn(paramtext):
	resp=get_named_entities(paramtext)
	#resp_format=format_response(resp)
	return resp

def main():
	val = get_named_entities("With the service number 12312412312 and name Mehmet Eris on 16/12/2015, we have recorded an incident about the Turk Telekom product.")
	resp_format=format_response(val)
	print(resp_format)
if __name__ == "__main__": main()
