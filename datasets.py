import pandas, numpy, re, os
from modules.learning.language_tools.tr_deasciifier import *
from core_helpers.helpers import *


def query_text(any_string):
    """
        Do not use pipe, search for each character in a text one by one.
        lil bit dummy but rgx pipe is much dummier.
        @Todo: Make this replace stuff a separate function
    """
    search_list = emoticons().keys()
    emo_dict = emoticons()
    emoticons_in_txt = []
    # if isinstance(any_string, list):
    #     for strings in any_string:
    #         for emo in search_list:
    #             pattern = "(" + emo.replace("(", "\\(").replace(")", "\\)") \
    #                 .replace("|", "\\|").replace("[", "\\[").replace("]", "\\]") + ")"
    #             m = re.findall(pattern, strings)
    #             if emo in m:
    #                 emoticons_in_txt.append(emo_dict[emo])
    #     return emoticons_in_txt

    for emo in search_list:
        pattern = "(" + emo.replace("(", "\\(").replace(")", "\\)")\
            .replace("|", "\\|").replace("[", "\\[").replace("]", "\\]") + ")"
        m = re.findall(pattern, any_string)
        if emo in m:
            emoticons_in_txt.append(emo_dict[emo])
    return emoticons_in_txt


def emoticons():
    """
    @Todo: Extend the list sometime.
    :return: emoticons
    """
    emoticons = {
        ':)': 'HAPPYFACE',
        ':]': 'HAPPYFACE',
        '.)': 'HAPPYFACE',
        '=)': 'HAPPYFACE',
        '(:': 'HAPPYFACE',
        ':‑)': 'HAPPYFACE',
        ':D': 'LAUGHINGFACE',
        '=D': 'LAUGHINGFACE',
        ':d': 'LAUGHINGFACE',
        '=d': 'LAUGHINGFACE',
        '.d': 'LAUGHINGFACE',
        ':-D': 'LAUGHINGFACE',
        ':(': 'UNHAPPYFACE',
        '=(': 'UNHAPPYFACE',
        ':[': 'UNHAPPYFACE',
        '.(': 'UNHAPPYFACE',
        '):': 'UNHAPPYFACE',
        ':-(': 'UNHAPPYFACE',
        ':\'(': 'CRYINGFACE',
        '=\'(': 'CRYINGFACE',
        ':\'[': 'CRYINGFACE',
        ':\'-(': 'CRYINGFACE',
        ':@': 'ANGRYFACE',
        '=@': 'ANGRYFACE',
        ':O': 'SHOCKEDFACE',
        ':o': 'SHOCKEDFACE',
        'o.o': 'SHOCKEDFACE',
        'O.o': 'SHOCKEDFACE',
        'o_O': 'SHOCKEDFACE',
        'O_o': 'SHOCKEDFACE',
        ':s': 'EMBRASEDFACE',
        ':S': 'EMBRASEDFACE',
        ' :$': 'EMBRASEDFACE',
        '=$': 'EMBRASEDFACE',
        ':/': 'CONFUSEDFACE',
        ':- *': 'KISSINGFACE',
        ': *': 'KISSINGFACE',
        ':×': 'KISSINGFACE',
        '#‑)': 'DRUNKFACE'
    }
    return emoticons


def tag_emoticons(any_string):
    search_list = emoticons().keys()
    emo_dict = emoticons()

    for emo in search_list:
        pattern = "(" + emo.replace("(", "\\(").replace(")", "\\)") \
            .replace("|", "\\|").replace("[", "\\[").replace("]", "\\]") + ")"
        m = re.findall(pattern, any_string)
        if emo in m:
            any_string = any_string.replace(emo, " "+ emo_dict[emo]+ " ")
    return any_string


def process(instances):
    # return tweetIsalnum(tweetIsnumeric(cleaner(instances)))
    return deasciify_word(cleaner(tag_emoticons(instances)))

def _get_data_from_xlsx(file_name, folder, X_name, y_name):
    path = folder + file_name
    veri = pandas.read_excel(path)
    instances = veri[X_name].values.tolist()
    instances = [str(i) for i in instances]
    labels = veri[y_name].values.tolist()
    instances = [process(i).strip() for i in instances]
    return instances, labels


def _get_data_from_csv(file_name, folder, X_name, y_name, sep='\t'):
    path = folder+file_name
    veri = pandas.read_csv(path, sep=sep)
    instances = veri[X_name].values.tolist()
    instances = [str(i) for i in instances]
    labels = veri[y_name].values.tolist()
    instances = [i.strip() for i in instances]
    return instances, labels


def _chech_extension(link):
    link = link.split('.')
    return link[-1]


def _get_and_chech_data(file_name, folder, X_name, y_name, sep='\t'):
    if _chech_extension(file_name):
        return _get_data_from_xlsx(file_name, folder, X_name, y_name)
    elif _chech_extension(file_name) == 'csv':
        return _get_data_from_csv(file_name, folder, X_name, y_name, sep=sep)


def get_all_data(inputs, mapping=False, sep='\t', show=False, n=False, return_type=True):
    try:
        if isinstance(inputs, list):
            data = []
            for row in inputs:
                instances, labels = _get_and_chech_data(row[0], row[1], row[2], row[3], sep)
                data.append(pandas.DataFrame({'text': instances, 'cat': labels}))
            veri = pandas.concat(data, ignore_index=True)
            veri['text'].replace('', numpy.nan, inplace=True)
            veri.dropna(subset=['text'], inplace=True)
            veri = veri.drop_duplicates(['text'])
            veri = veri.reset_index()

            if show:
                print('Value Counts :')
                print(veri.cat.value_counts())

            if mapping:
                veri.cat = veri.cat.map(mapping)

            if n:

                veri = shuffle(veri.text, veri.cat, return_type=return_type)
                if n == 'max':
                    n = veri.cat.value_counts().min()
                a = veri[veri['cat'] == 'positive'].sample(n=n)
                b = veri[veri['cat'] == 'negative'].sample(n=n)
                c = veri[veri['cat'] == 'neutral'].sample(n=n)
                veri = pandas.concat([a, b, c], names=['text', 'cat'], ignore_index=True)

            print('Labels \t\t : ', veri.cat.unique())
            print('Value Counts : ')
            print(veri.cat.value_counts())
            instances = veri['text'].tolist()
            liste = []

            for instance in instances:
                liste.append(instance)

            instances = liste
            labels = veri.cat.tolist()

            return instances, labels

    except:
        print("Deniyoruz Canım ama olmuyor.")


def cleaner(data, mapping=False):
    if not mapping:
        mapping={
            '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})': 'URLs',
            '(?:\@+[\w_]+[\w\'_\-]*[\w_]+)': ' ',  # Mention ..
            '(?:\#+[\w_]+[\w\'_\-]*[\w_]+)': ' ',  # Hashtag..
            r'([a-z])\1+': r'\1',  # remove repeat letter
            '(?:^|\W)rt': ' ',  # Remove spesific word rt
            '(?:^|\W)at': ' ',  # Remove spesific word at
            '(?:^|\s):': ' ',  # ikinokta üstü üste şeysi..
            '(?:[\W_]+)': ' ',  # punctuation (:,.{[(%&+ ..
            "(?:[0-9'\-_])":" ",
        }
    harfler = {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}
    if isinstance(data, list):
        data2 = []
        for row in data:
            row=row.lower()
            for key, value in harfler.items():
                row=re.sub(key, value, row)
            for key, value in mapping.items():
                row=re.sub(key, value, row)
            data2.append(row.strip())
        return data2

    if isinstance(data, str):
        data=data.lower()
        for key, value in harfler.items():
            data = re.sub(key, value, data)
        for key, value in mapping.items():
            data=re.sub(key, value, data)
        return data.strip()


def tweetIsnumeric(instances):
    liste = []
    if isinstance(instances, list):
        for cumle in instances:
            liste.append(' '.join(kelime for kelime in cumle.split() if not kelime.isnumeric()))
        return liste
    elif isinstance(instances, str):
        return(' '.join(kelime for kelime in instances.split() if not kelime.isnumeric()))
    else:
        pass


def tweetIsalnum(instances):
    liste = []
    if isinstance(instances, list):
        for cumle in instances:
            liste.append(' '.join(kelime for kelime in cumle.split() if kelime.isalnum()))
        return liste
    elif isinstance(instances, str):
        return(' '.join(kelime for kelime in instances.split() if kelime.isalnum()))
    else:
        pass


def bring_data(data_set_name, n):
    if data_set_name == 'SENTIMENT':
        unique_labels       = bring_sentiment_labels()
        paths_and_labels    = bring_sentiment_data_paths()
        instances, labels = get_all_data(paths_and_labels, mapping=unique_labels, n=n)
        return instances, labels


def shuffle(instances, labels, return_type):

    if not isinstance(instances, pandas.Series) and not isinstance(labels, pandas.Series):
        instances=pandas.Series(instances)
        labels=pandas.Series(labels)

    numpy.random.seed(10)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(labels)))
    data_shuffled = instances[shuffle_indices]
    label_shuffled = labels[shuffle_indices]
    # print(label_shuffled.value_counts())

    if return_type:
        data = pandas.DataFrame({'text':data_shuffled, 'cat':label_shuffled})
        return data

    return data_shuffled, label_shuffled
