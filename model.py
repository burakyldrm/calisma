import modules.learning.text_categorization.prototypes.classification.classification_system as clsf_sys
import modules.learning.text_categorization.prototypes.tasks.sentiment_analyser as sa_task
# from b_func import *
from datasets import *
import model_config
from core_helpers import model_helpers as mh

if __name__ == '__main__':
    instances, labels = bring_data("SENTIMENT", "max")



    test_instances = [
        "senin ananı sikerim",
        "merhaba size nasıl yardımcı olabilirim",
        "sen ne ibnetorsun öyle",
        "bana müdürü bağla",
        "bana müdürü bağlar mısın ?",
        "ibnesin oğlum sen",
        "eve gitmek istiyorum ama çok trafik var",
        "seni seviyorum ama asla!",
        "merhaba size nasıl yardımcı olabilirim",
        "Sen ne iyisin",
        "Garip bir durum",
        "Fulya çok mu tatlı acaba yaaa",
        "maçta içinizden geçtik resmen",
        "Recep Tayyip Erdoğan",
        "Fethullah gülen",
        "fetö",
        "feto"
    ]

    modelrootfolder = '/home/burak/Desktop'
    modelname = 'burakV4'


    model, modelfolder = mh.model(mh.system("TR Sentiment Analysis"), instances, labels, 10, modelrootfolder, modelname)

    modelfolder = os.path.join(modelrootfolder, modelname)
    labels, maps, a = mh.predict(mh.system("TR Sentiment Analysis"), modelfolder, test_instances)
    print(labels)


    # # #
    # # #
    # # #
    # tr_sentiment = clsf_sys.ClassificationSystem(
    #     Task=sa_task.SentimentAnalysis,
    #     task_name="TR Sentiment Analysis",
    #     config_obj=model_config.tr_config_object
    # )
    #
    # accuracy, fscore, duration, root = tr_sentiment.get_cross_validated_clsf_performance(instances, labels, nfolds=7)
    #
    # modelrootfolder = '/home/burak/Desktop'
    # modelname = 'burakV3'
    #
    # model, modelfolder = tr_sentiment.train_and_save_model(instances, labels, modelrootfolder, modelname)
    #
    # modelfolder = os.path.join(modelrootfolder, modelname)
    # # modelfolder = modelrootfolder + "/" + modelname
    #
    # test_instances = [
    #     "senin ananı sikerim",
    #     "merhaba size nasıl yardımcı olabilirim",
    #     "sen ne ibnetorsun öyle",
    #     "bana müdürü bağla",
    #     "bana müdürü bağlar mısın ?",
    #     "ibnesin oğlum sen",
    #     "eve gitmek istiyorum ama çok trafik var",
    #     "seni seviyorum ama asla!",
    #     "merhaba size nasıl yardımcı olabilirim",
    #     "Sen ne iyisin",
    #     "Garip bir durum",
    #     "Fulya çok mu tatlı acaba yaaa",
    #     "maçta içinizden geçtik resmen",
    #     "Recep Tayyip Erdoğan",
    #     "Fethullah gülen",
    #     "fetö",
    #     "feto"
    # ]
    #
    # predicted_labels, prediction_map, a = tr_sentiment.predict_offline(modelfolder, mh.deasciify_test(test_instances))
    # print(predicted_labels)