def bring_sentiment_labels():
    mapping = {
        'ilgisiz': 'neutral',
        'Positive': 'positive',
        'positive': 'positive',
        'Negative': 'negative',
        'negative': 'negative',
        'Neutral': 'neutral',
        'neutral': 'neutral',
        'Karar Veremedim': 'neutral',
        'Olumsuz': 'negative',
        'olumsuz': 'negative',
        'Olumlu': 'positive',
        'olumlu': 'positive',
        'Nötr': 'neutral',
        'nötr': 'neutral'
    }
    return mapping


def bring_sentiment_data_paths():
    veriler = [
        ['disagreed.xlsx',                  '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['sentiment_agreed_skip.xlsx',      '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['sentiment_agreed_neutral_v2.xlsx','/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['tr_polartweets.xlsx',             '/home/burak/Desktop/Sentiment/data/', 'text', 'polarity'],
        ['sentiments_logs_labeled.xlsx',    '/home/burak/Desktop/Sentiment/data/', 'Content', 'Category'],
        ['negative.xlsx',                   '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['positive.xlsx',                   '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['kufur.xlsx',                      '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['argo.xlsx',                       '/home/burak/Desktop/Sentiment/data/', 'Content', 'new_cat'],
        ['positive_film.xlsx',              '/home/burak/Desktop/Sentiment/data/', 'text', 'new_cat'],
        ['y_yorum.xlsx',                    '/home/burak/Desktop/Sentiment/data/', 'text', 'new_cat'],
        ['neutral.xlsx',                    '/home/burak/Desktop/Sentiment/data/', 'text', 'new_cat'],
        ['neutral_turkceler.xlsx',          '/home/burak/Desktop/Sentiment/data/', 'veriler', 'new_cat'],
    ]

    return veriler