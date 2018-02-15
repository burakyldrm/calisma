"""
spell_check_eng.py
Company: Etiya
Author: Ali Cagatay Kuru
Last Check Date: 03.04.2014
"""

from autocorrect import spell
from nltk.tokenize import sent_tokenize, word_tokenize
import re

#For text correction choice
def correctText(paramtext):
	resulting_text = ""
	pattern = re.compile(u'[\w]+')
	resulting_text = pattern.sub(correctMatch, paramtext)	
	
	return resulting_text

#For word correction choice
def correctWord(paramtext):
	return spell(paramtext)

def correctMatch(match):
	word = str(match.group())
	try:
		int(word)
		corrected_word = word
	except ValueError:
		corrected_word = spell(word)	
	return corrected_word
