# -*- coding: utf-8 -*-

from polyglot.text import Text
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import datetime
import time
import io

def get_named_entities(text, lang="tr"):
	
	aylar = ['', 'ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran', 'temmuz', 'ağustos', 'eylül', 'ekim', 'kasım', 'aralık']

	sents_for_hizmet = sent_tokenize(text)
	h_num = []
	hizmet_nums = []

	compiled_re_for_date = re.compile(r'(\d{2})[/.-](\d{2})[/.-](\d{4})$')
	compiled_re_for_clock = re.compile(r'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$')

	dates = []
	dates2 = []
	date_manuel = ""

	boolean_for_lemmatization = 0

	clocks = []
	clocks2 = []

	ctr = 0		
	for sent in sents_for_hizmet:
		words = word_tokenize(sent)
		for w in words:
			if test_tarih(w, compiled_re_for_date) == 1:
				dates.append(w)
			word = w.lower()

			if word in aylar:
				findDates(word, words, ctr, dates)

			test_clock(w, compiled_re_for_clock, clocks)
			#print(w)

			if test_hizmet_numarasi(w)==0:
				h_num = []
			else:
				hizmet_nums.append(w)

			for i in range(len(w)):
				if w[i] == "'":
					text = text + " " + w[:i]
					boolean_for_lemmatization = 1

			if (w[len(w) -2:] == "da" or w[len(w) -2:] == "de") and boolean_for_lemmatization == 0:
				text = text + " " + w[:len(w) -2]

			ctr = ctr + 1
			boolean_for_lemmatization = 0
		ctr = 0
	'''	
	for nums in hizmet_nums:
		if isValidTCID(nums) == 1:
			h_num.extend([("".join("T.C.Kimlik Num: "), nums)])
		else:
			h_num.extend([("".join("Hizmet Num: "), nums)])
	'''
	'''
	for date in dates:
	'''
	'''
		try:
			unicode(date, "ascii")
		except UnicodeError:
			date = unicode(date, "utf-8")
		else:
			# value was valid ASCII data
			pass
		'''
		#dates2.extend([("".join("Tarih: "), date)])
	'''
	for clock in clocks:
		clocks2.extend([("".join("Saat: "), clock)])
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

def test_tarih(datestring, pattern_date):
        try:
                mat=re.match(pattern_date, datestring)
                if mat is not None:
                        datetime.datetime(*(map(int, mat.groups()[-1::-1])))
                        return 1
        except ValueError:
                pass
        return 0

def isValidTCID(value):
	value = str(value)

	# TC kimlik 11 haneli, rakamlardan oluşan ve 0 ile başlamayan sayıdır.
	if not len(value) == 11 or not value.isdigit() or int(value[0]) == 0:
	    return -1

	digits = [int(d) for d in str(value)]

	if not sum(digits[:10]) % 10 == digits[10]:
	    return 0

	if not (((7 * sum(digits[:9][-1::-2])) - sum(digits[:9][-2::-2])) % 10) == digits[9]:
	    return 0
	# Butun kontrollerden gecti.
	return 1

def test_clock(str, pattern_clock, clocks):
	found_bool = re.match(pattern_clock, str)
	if found_bool is None:
		pass
	else:
		clocks.append(str)

def findDates(word, words, ctr, dates):
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
	

def getNamedEntityTr(paramtext):
	resp=get_named_entities(paramtext)
	#resp_format=format_response(resp)
	return resp
	
def main():

	val=get_named_entities("1 kAsım 2017 23:45 tarihinde başvurmuş 8800550672 hizmet numaralı Türk Telekom müşterisi olan ve bu devrede yaşanan Ankara'da, Eskişehirde İstanbulda problem için 26554689072 Tc kimlik numaralı müşteri ile ilgili olarak ekiplerle 23.01.2017 17:30:45 Tarihinde irtibata geçilerek tarafınıza bilgi iletilecektir.")
	res=format_response(val)
	print (res)

if __name__ == "__main__": main()
