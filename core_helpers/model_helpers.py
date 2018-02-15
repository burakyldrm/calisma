from modules.learning.language_tools.tr_deasciifier import *
import modules.learning.text_categorization.prototypes.classification.classification_system as clsf_sys
import modules.learning.text_categorization.prototypes.tasks.sentiment_analyser as sa_task
import model_config
import os



def deasciify_test(test_instances):
    new_test = []
    for instance in test_instances:
        new_test.append(deasciify_word(instance))
    return new_test



def system(task_name):
    return clsf_sys.ClassificationSystem(
        Task=sa_task.SentimentAnalysis,
        task_name=task_name,
        config_obj=model_config.tr_config_object
    )


def model(tr_sentiment, instances, labels, nfolds, rootfolder, modelname):
    # tr_sentiment=system(task_name)
    tr_sentiment.get_cross_validated_clsf_performance(instances, labels, nfolds)
    model, modelfolder = tr_sentiment.train_and_save_model(instances, labels, rootfolder, modelname)
    # modelfolder = os.path.join(rootfolder, modelname)
    return model, modelfolder


def predict(tr_sentiment, modelfolder, test_instances):
    return tr_sentiment.predict_offline(modelfolder, deasciify_test(test_instances))