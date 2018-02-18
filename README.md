# SENTIMENT ANALYSIS
## Projeye Eklenmesi

Projeyi indirdikten sonra Ana Klasöre "Library" isminde bir klasör oluşturulması gerekir.
Bu klasörün içerisine Dicle'nin yazdığı kodlardan "modules" isimli klasör kopyalanır.

Dosya yapısı aşağıdaki gibi olmalıdır.
```
sentiment
   |-- cache/
   |-- core_helpers/
   |-- data/
   |-- doc/
   |-- logs/ 
   |-- library/
       |-- modules/
   |-- datasets.py
   |-- model_config.py
   |-- README.md
   |-- requirements.txt
   |-- model.py
```

## Modelin Test Edilmesi

Modeli test etmek için model.py dosyası içerisindeki "test_instances" değişkenine herhangi bir cümle ekleyerek, cümlenin sentiment analizi yapılabilir. 

## Modelin Train edilmesi

Modeli yeni baştan train etmek için, Yorum satırına alınan;
   
``` 
model, modelfolder= mh.model(mh.system("TR Sentiment Analysis"), instances, labels, 10, modelrootfolder, modelname)
```

kod satırı açılır ve model.py dosyası çalıştırılır.

## Modele yeni veriler eklemek

Modele yeni veriler eklemek için öncelikle data klasörü içerisine veriler eklenir. Daha sonra core_helpers klasörü içerisindeki helpers.py dosyasında "bring_sentiment_data_paths()" fonksiyonu içerisine veri yolu ve kullanılacak kolonların isimleri ile yazılır. Örneğin;

```
['deneme.xlsx', '../data_klasor/', 'text_col_name', 'category_col_name']
```
