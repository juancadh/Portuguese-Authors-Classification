import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import eli5
import warnings
warnings.filterwarnings(action = 'ignore')

class classification_battery():
    """
    ************ CLASSIFICATION BATTERY **************

    Battery of different methods that help to build and test some classifications for text.

    ========= EXAMPLE =========

    Xfeatures = df["quotes"]  # ["We write to taste life twice, in the moment and in retrospect.", 
                              #  "One day I will find the right words, and they will be simple."
                              #  "Either write something worth reading or do something worth writing."
                              #  ... ]
    ylabels   = df["author"]  # ["Anaïs Nin", "Jack Kerouac", "Benjamin Franklin", ...]
    config = {
            "print"         : True,
            "lang"          : "english",
            "test_size"     : 0.2,
            "vectorization" : "tf-idf",   # "tf-idf" "bag-of-words" "hashing"
            "n_gram"        : "word",     # 'word', 'char', 'char_wb'
            "n_range"       : (1,1),      # ex: (1,1), (1,2), (1,3), (1,4), (1,5), (2,3)
            "metric_avg"    : "weighted",  # None, binary (default), micro, macro, samples, weighted
            "classifiers"   : ["NB", "logit", "randomForest", "SVM", "SGDC", "DecTree"],
            "cross_validate" : False
        }

    cls_bat = classification_battery(Xfeatures, ylabels, config)

    # Run KNN baseline with k_range up to 10.
    knn_mean_error, metrics_perf = cls_bat.baseline(k_rng = 10)

    # Run the baseline with the best performance according to the previous line and save the model
    knn_mean_error, metrics_perf, model_spec_baseline = cls_bat.baseline(specific= 3)

    # Run the search for the best model using those possibilities
    df_perf = cls_bat.search_models(vect_list = ['tf-idf'], 
                                analizers  = ['word', 'char', 'char_wb'], 
                                n_rang_lst = [(1,5), (2,5), (3,5), (1,6), (1,7)])

    # Check the previuos results sorted by the metric LOG_F1
    df_perf.sort_values("LOG_F1", ascending = False).reset_index(drop = True).head()

    # Run the best model
    config = {
            "print"         : True,
            "lang"          : "english",
            "test_size"     : 0.2,
            "vectorization" : "tf-idf",  
            "n_gram"        : "char",    
            "n_range"       : (1,7),
            "metric_avg"    : "weighted",
            "classifiers"   : ["NB", "logit", "randomForest", "SVM", "SGDC", "DecTree"],
            "cross_validate" : True
        }
    best_model = classification_battery(Xfeatures, ylabels, config)
    best_model_sel = best_model.run_models()

    # Test the model predicting one out-of-sample sentence
    sample = ["You never have to change anything you got up in the middle of the night to write."]
    ts = best_model.predict_sample(sample, cv_list, model_list, labels)

    """

    cv_list    = []
    model_list = []
    model_lbl  = []

    def __init__(self, X_d, y_d, config = {}):

        self.config = config

        if "print" not in config:
            self.config["print"] = True
        if "lang" not in config:
            self.config["lang"] = "english"
        if "test_size" not in config:
            self.config["test_size"] = 0.2
        if "vectorization" not in config:
            self.config["vectorization"] = "tf-idf"
        if "n_gram" not in config:
            self.config["n_gram"] = "word"
        if "n_range" not in config:
            self.config["n_range"] = (1,2)
        if "metric_avg" not in config:
            self.config["metric_avg"] = "binary"
        if "classifiers" not in config:
            self.config["classifiers"] = ["NB", "logit", "randomForest", "SVM", "SGDC"]
        if "cross_validate" not in config:
            self.config["cross_validate"] = False


        self.X_data = X_d
        self.y_data = y_d

        self.labels_y = list(pd.Series(y_d).unique())

        self.set_vectorizer()

    # ========================================================================================================================
    # ========================================================================================================================
    
    def set_vectorizer(self):

        _max_features = 3000 # Maximium number of features (best 1500)
        _max_df       = 0.8  # Ignore terms that have a document frequency more than 80% (domain specific) 
        _min_df       = 5    # Ignore terms that have a document frequency lower than 5.

        # Vectorize the data
        if self.config["vectorization"] == "bag-of-words":
            # BAG OF WORDS VECTORIZER
            if self.config["n_gram"] != "":
                self.cv = CountVectorizer(analyzer   = self.config["n_gram"],
                                        ngram_range  = self.config["n_range"],
                                        max_features = _max_features, 
                                        max_df       = _max_df,
                                        min_df       = _min_df,
                                        stop_words   = stopwords.words(self.config["lang"])
                                    )
            else:
                self.cv = CountVectorizer(max_features = _max_features,
                                        max_df         = _max_df,
                                        min_df         = _min_df,
                                        stop_words     = stopwords.words(self.config["lang"])
                                    )

        elif (self.config["vectorization"] == "tf-idf"):
            # TF-IDF VECTORIZER
            if self.config["n_gram"] != "":
                self.cv = TfidfVectorizer(analyzer = self.config["n_gram"],
                                        ngram_range = self.config["n_range"]
                                    ) 
            else:
                 self.cv = TfidfVectorizer(max_features = _max_features,
                                        max_df=_max_df,
                                        min_df=_min_df,
                                        stop_words=stopwords.words(self.config["lang"])
                                    )
        elif (self.config["vectorization"] == "hashing"):
            # (TEST) HASHING VECTORIZATION
            if self.config["n_gram"] != "":
                self.cv = HashingVectorizer(analyzer = self.config["n_gram"], 
                                        ngram_range = self.config["n_range"]
                                    )
            else:
                self.cv = HashingVectorizer()
        else:
            print("Default vectorization used: Bag-of-Words")
            self.cv = CountVectorizer(max_features = 1500,
                                    max_df=0.8,
                                    min_df=5,
                                    stop_words=stopwords.words(self.config["lang"])
                                )
    # ========================================================================================================================
    # ========================================================================================================================
        
    def transform_label(self, serie_lst, label_lst = []):
        """ Function that transform the categorical values of serie_lst into numerical values """
        if len(label_lst) > 0:
            return [label_lst.index(i) for i in serie_lst], label_lst
        else:
            label_lst = list(pd.Series(serie_lst).unique())
            return [label_lst.index(i) for i in serie_lst], label_lst

    # ========================================================================================================================
    # ========================================================================================================================
    
    def test_feature(self, df, variable, col_label):
        """ Test if the feature is good or not to predict. 
            df        : Dataframe with labels and features
            variable  : name of column name of the feature
            col_label : name of column name of the label target.

            ex: plt = test_feature(df, 'num_emojis', 'person')
        """
        for i in df[col_label].unique():
            plt.hist(df[df[col_label]==i][variable], alpha = 0.5, normed=True, label = i)
        plt.legend(loc = 'upper right')
        plt.title(f"Test of feature: {variable}")
        return plt

    # ========================================================================================================================
    # ========================================================================================================================
    
    def baseline(self, k_rng = 10, specific = 0):
        """
            Function that runs the baseline of the model.
        """

        Xfeatures = self.X_data
        ylabels   = self.y_data
        X  = self.cv.fit_transform(Xfeatures)

        # Split the database
        x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size = self.config["test_size"], random_state = 42)

        y_test_trans, lbls = self.transform_label(y_test, label_lst = self.labels_y)

        knn_mean_error     = []
        accuracy_lst       = []
        f1_lst             = []
        recall_lst         = []
        precision_lst      = []
        model_spec     = {}

        if specific > 0:
            print(f"Specific value {specific} has been selected.")
            knn = KNeighborsClassifier(n_neighbors=specific)
            acc_val, f1_val, recall_val, precision_val, y_pred, y_pred_trans = self.train_model(knn, x_train, y_train, x_test, y_test)
            print(f"\nBaseline - KNN({specific})")
            print(f"Accuracy: {round(acc_val,3)} | F1: {round(f1_val,3)} | Recall: {round(recall_val,3)} | Precision: {round(precision_val,3)}")
            #print(f"Accuracy: {round(accuracy,3)}")
            
            knn_mean_error.append(np.mean(y_pred != y_test))
            accuracy_lst.append(acc_val)
            f1_lst.append(f1_val)
            recall_lst.append(recall_val)
            precision_lst.append(precision_val)

            metrics_perf = pd.DataFrame(np.column_stack([accuracy_lst, f1_lst, recall_lst, precision_lst]))
            metrics_perf.columns = ["accuracy","f1","recall","precision"]

            model_spec['KNN'] = {'cv': self.cv, 'model': knn}

            self.model_baseline = model_spec
            return knn_mean_error, metrics_perf, model_spec

        else:
            # Calculating error for K values between 1 and 40
            for i in range(1, k_rng+1):
                knn = KNeighborsClassifier(n_neighbors=i)
                acc_val, f1_val, recall_val, precision_val, y_pred, y_pred_trans = self.train_model(knn, x_train, y_train, x_test, y_test)
                print(f"\nBaseline - KNN({i})")
                print(f"Accuracy: {round(acc_val,3)} | F1: {round(f1_val,3)} | Recall: {round(recall_val,3)} | Precision: {round(precision_val,3)}")
                #print(f"Accuracy: {round(accuracy,3)}")

                knn_mean_error.append(np.mean(y_pred != y_test))
                accuracy_lst.append(acc_val)
                f1_lst.append(f1_val)
                recall_lst.append(recall_val)
                precision_lst.append(precision_val)

            k_lst = range(1, k_rng+1)
            metrics_perf = pd.DataFrame(np.column_stack([k_lst,accuracy_lst, f1_lst, recall_lst, precision_lst]))
            metrics_perf.columns = ["K","accuracy","f1","recall","precision"]
            metrics_perf.set_index("K", inplace=True)

            plt.figure(figsize=(12, 5))
            plt.plot(k_lst, knn_mean_error, color='#333333', marker='o', markerfacecolor='#FFFFFF', markersize=7, label = "Mean Error")
            plt.title('Mean Error')
            plt.xlabel('K Value')
            plt.legend(loc = "upper left")
            plt.show()

            plt.figure(figsize=(12, 5))
            plt.bar(k_lst, accuracy_lst, color='#FFBA00', label = "Accuracy")
            plt.bar(k_lst, f1_lst, color='#FF6347', label = "F1")
            plt.bar(k_lst, recall_lst, color='#F8B195', label = "Recall")
            plt.bar(k_lst, precision_lst, color='#6C5B7B', label = "Precision")
            plt.title('Out-of-Sample Performace')
            plt.xlabel('K Value')
            plt.legend(loc = "best")
            plt.show()

            return knn_mean_error, metrics_perf

        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))

    # ========================================================================================================================
    # ========================================================================================================================

    def train_model_crossval(self, classifier, X, y, n_folds = 5, metrics = []):
        """
        Train the classifier using cross-validation.
        Splitting the data, fitting a model and computing the score <n_folds> consecutive times (with different splits each time).
        """
        
        # Fit the classifier, for future predictions
        classifier.fit(X, y)

        if len(metrics)==0:
            if self.config["metric_avg"] == 'binary':
                metrics = ['accuracy', 'f1', 'precision', 'recall']
            else:
                metrics = ['accuracy', 'f1_'+self.config["metric_avg"], 'precision_'+self.config["metric_avg"], 'recall_'+self.config["metric_avg"]]

        # Cross Validation Scores
        results = {}

        # Metrics
        for i in metrics:
            results[i] = {}
            results[i]['scores'] = cross_val_score(classifier, X, y, cv=n_folds, scoring=i)
            results[i]['mean']   = results[i]['scores'].mean()
            results[i]['std']    = results[i]['scores'].std() * 2

        return results, metrics

    # ========================================================================================================================
    # ========================================================================================================================

    def train_model(self, classifier, x_train, y_train, x_test, y_test):

        # Fit the training dataset on the classifier
        classifier.fit(x_train, y_train)

        # Predict labels in the validation dataset
        y_pred = classifier.predict(x_test)

        # Transform labels from catagorical to numerical representation
        y_test_trans, lbls = self.transform_label(y_test, label_lst = self.labels_y)
        y_pred_trans, lbls = self.transform_label(list(y_pred), label_lst = self.labels_y)

        # Metrics of performance
        acc_val       = accuracy_score(y_test_trans, y_pred_trans)
        f1_val        = f1_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])
        recall_val    = recall_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])
        precision_val = precision_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])

        return acc_val, f1_val, recall_val, precision_val, y_pred, y_pred_trans

    # ========================================================================================================================
    # ========================================================================================================================

    def compute_classifier(self, classifier, X, ylabels, x_train, y_train, x_test, y_test, class_name, round_dig = 5):

        if self.config["cross_validate"]:
            crossval_scores, metrics = self.train_model_crossval(classifier, X, ylabels, n_folds = 5, metrics = [])
            acc_val        = crossval_scores['accuracy']['mean']
            f1_val         = crossval_scores['f1_' + self.config["metric_avg"]]['mean']
            recall_val     = crossval_scores['recall_' + self.config["metric_avg"]]['mean']
            precision_val  = crossval_scores['precision_' + self.config["metric_avg"]]['mean']
        else:
            acc_val, f1_val, recall_val, precision_val, y_pred, y_pred_trans = self.train_model(classifier, x_train, y_train, x_test, y_test)
        
        if self.config['print']:
            if self.config["cross_validate"]:
                print(f"\n{class_name} <cross-validation>")
            else:
                print(f"\n{class_name}")
            print(f"Accuracy: {round(acc_val,round_dig)} | F1: {round(f1_val,round_dig)} | Recall: {round(recall_val,round_dig)} | Precision: {round(precision_val,round_dig)}")

        return {'cv' : self.cv, 'model' : classifier, "accuracy" : acc_val, "f1" : f1_val, "recall" : recall_val, "precision" : precision_val}
        
    # ========================================================================================================================
    # ========================================================================================================================

    def run_models(self, round_dig=5):
        """
        This function performs different classification models in order to check the best accuracy.
        """

        Xfeatures = self.X_data
        ylabels   = self.y_data

        # Transform the features into a numerical representation (Bag-Of-Words, TF-IDF, HashingVectorizer)
        X  = self.cv.fit_transform(Xfeatures)

        model_spec = {}

        # *** 2. Split Database
        x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size = self.config["test_size"], random_state = 42)
        #y_test_trans, lbls = self.transform_label(y_test, label_lst = self.labels_y)
        
        # *** 3. Models
        # --------------------------------
        # ------ Naive Bayes Model -------
        # --------------------------------
        if 'NB' in self.config["classifiers"]:
            clf = MultinomialNB()
            model_spec['NB'] = self.compute_classifier(clf, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Naive Bayes", round_dig = round_dig)

        # --------------------------------
        # ----- Logistic Model -----------
        # --------------------------------
        if 'logit' in self.config["classifiers"]:
            logit = LogisticRegression()
            model_spec['logit'] = self.compute_classifier(logit, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Logistic", round_dig = round_dig)

        # --------------------------------------
        # ----- Random Forest Model -----------
        # -------------------------------------
        if 'randomForest' in self.config["classifiers"]:
            randForest = RandomForestClassifier()
            model_spec['randomForest'] = self.compute_classifier(randForest, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Random Forest", round_dig = round_dig)

        # --------------------------------------------------
        # ----- Support Vector Machine Classifier -----------
        # --------------------------------------------------
        if 'SVM' in self.config["classifiers"]:
            svmc = SVC()
            model_spec['SVM'] = self.compute_classifier(svmc, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Support Vector Machine", round_dig = round_dig)
 
        # ------------------------------------------------
        # ---------- Decision Tree Classifier ------------
        # ------------------------------------------------
        if 'DecTree' in self.config["classifiers"]:
            decTree = DecisionTreeClassifier(max_depth=3)
            model_spec['DecTree'] = self.compute_classifier(decTree, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Decision Tree", round_dig = round_dig)
 
        # --------------------------------------------------------------
        # ----- Stochastic Gradient Descent (SGD) Classifier -----------
        # --------------------------------------------------------------
        if 'SGDC' in self.config["classifiers"]:
            sgdc = SGDClassifier(random_state=42)
            model_spec['SGDC'] = self.compute_classifier(sgdc, X, ylabels, x_train, y_train, x_test, y_test, class_name = "Stochastic Gradient Descent (SGD)", round_dig = round_dig)
 
        # -------------------------------------------------
        # ----------- Neural Network ----------------------
        # -------------------------------------------------
        # TODO: In the future!


        self.models = model_spec
        return model_spec

    # ========================================================================================================================
    # ========================================================================================================================
    
    def predict_sample(self, sample, printIt = True):
        """ Function that tests a sample of the text you want to classify given a Vectorizer list and a Model list. Labels are the names of the models to be printed."""
        cv_list    = []
        model_list = []
        labels     = []

        if hasattr(self, 'model_baseline') and hasattr(self, 'models'):
            cv_list.append(self.model_baseline['KNN']['cv'])
            model_list.append(self.model_baseline['KNN']['model'])
            labels.append('KNN (BASELINE)')
            for i in self.config['classifiers']:
                cv_list.append(self.models[i]['cv'])
                model_list.append(self.models[i]['model'])
                labels.append(i)
        elif hasattr(self, 'model_baseline'):
            cv_list.append(self.model_baseline['KNN']['cv'])
            model_list.append(self.model_baseline['KNN']['model'])
            labels.append('KNN (BASELINE)')
        elif hasattr(self, 'models'):
            for i in self.config['classifiers']:
                cv_list.append(self.models[i]['cv'])
                model_list.append(self.models[i]['model'])
                labels.append(i)
        else:
            print("No model or baseline has been run. Use run_models() or baseline() methods before test sample.")
            return []  

        predictions = []
        for i in range(len(cv_list)):
            vect = cv_list[i].transform(sample).toarray()
            pred = model_list[i].predict(vect)[0]
            predictions.append((labels[i], pred))
            if printIt:
                print(f"{labels[i]}: ", pred)

        return predictions

    # ========================================================================================================================
    # ========================================================================================================================
    
    def search_models(self, vect_list = ["bag-of-words", "tf-idf"], analizers  = ['word', 'char', 'char_wb'], n_rang_lst = [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],  round_dig = 3):
        """ AAA """

        Xfeatures = self.X_data
        ylabels   = self.y_data
        X  = self.cv.fit_transform(Xfeatures)

        tot = len(vect_list) * len(analizers) * len(n_rang_lst)

        out_perf_F1 = {}
        out_perf_AC = {}
        for i in self.config['classifiers']:
            out_perf_F1[i] = []
            out_perf_AC[i] = []

        out_perf_F1['Vectorizer'] = []
        out_perf_AC['Vectorizer'] = []
        out_perf_F1['Analizer']   = []
        out_perf_AC['Analizer']   = []
        out_perf_F1['n_range']    = []
        out_perf_AC['n_range']    = []

        c = 1
        for vectorizer_i in vect_list:
            for analizer in analizers:
                for n_rang in n_rang_lst:
                    config = {
                            "print"          : False,
                            "lang"           : self.config['lang'],
                            "test_size"      : self.config['test_size'],
                            "vectorization"  : vectorizer_i,    # "tf-idf" "bag-of-words" "hashing"
                            "n_gram"         : analizer,        # 'word', 'char', 'char_wb'
                            "n_range"        : n_rang,          # (1,1) -> Only unigram, (1,2) -> Unigram and bigram, (1,3), (1,4), (1,5), (2,3)
                            "metric_avg"     : self.config['metric_avg'],
                            "classifiers"    : self.config['classifiers'],
                            "cross_validate" : self.config['cross_validate']
                        }

                    cls_bat    = classification_battery(Xfeatures, ylabels, config)
                    model_spec = cls_bat.run_models()

                    out_perf_F1['Vectorizer'].append(vectorizer_i)
                    out_perf_F1['Analizer'].append(analizer)
                    out_perf_F1['n_range'].append(n_rang)
                    out_perf_AC['Vectorizer'].append(vectorizer_i)
                    out_perf_AC['Analizer'].append(analizer)
                    out_perf_AC['n_range'].append(n_rang)

                    if self.config['cross_validate']:
                        txtCrsv = " | Using Cross-validation"
                    else:
                        txtCrsv = ""

                    print(f"\n=================== {c} of {tot} ======================")
                    print(f"Vectorizer: {vectorizer_i} | Analizer: {analizer} | n_range: {n_rang}{txtCrsv}\n")
                    txtAcc = "ACCURACY ====> "
                    txtF1 = "F1       ====> "
                    for i in self.config['classifiers']:
                        txtAcc += f"{i}:{round(model_spec[i]['accuracy'],round_dig)} | "
                        txtF1  += f"{i}:{round(model_spec[i]['f1'],round_dig)} | "
                        out_perf_AC[i].append(round(model_spec[i]['accuracy'],round_dig))
                        out_perf_F1[i].append(round(model_spec[i]['f1'],round_dig))

                    print(txtAcc)
                    print(txtF1)

                    c += 1

        out_perf_AC = pd.DataFrame(out_perf_AC)
        out_perf_F1 = pd.DataFrame(out_perf_F1)
        
        return out_perf_AC, out_perf_F1


"""
logit.fit(x_train, y_train)
y_pred = logit.predict(x_test)
y_pred_trans, lbls = self.transform_label(list(y_pred), label_lst = self.labels_y)
acc_val       = accuracy_score(y_test_trans, y_pred_trans)
f1_val        = f1_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])
recall_val    = recall_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])
precision_val = precision_score(y_test_trans, y_pred_trans, average=self.config["metric_avg"])



if self.config["cross_validate"]:
                crossval_scores, metrics = self.train_model_crossval(clf, X, ylabels, n_folds = 5, metrics = [])
                cv_acc_val        = crossval_scores['accuracy']['mean']
                cv_f1_val         = crossval_scores['f1_'+self.config["metric_avg"]]['mean']
                cv_recall_val     = crossval_scores['recall_'+self.config["metric_avg"]]['mean']
                cv_precision_val  = crossval_scores['precision_'+self.config["metric_avg"]]['mean']
            else:
                acc_val, f1_val, recall_val, precision_val, y_pred, y_pred_trans = self.train_model(clf, x_train, y_train, x_test, y_test)
             
            if self.config['print']:
                if self.config["cross_validate"]:
                    print("\nNaive Bayes model <cross-validation>")
                    print(f"CV-Accuracy: {round(cv_acc_val,round_dig)} | CV-F1: {round(cv_f1_val,round_dig)} | CV-Recall: {round(cv_recall_val,round_dig)} | CV-Precision: {round(cv_precision_val,round_dig)}")
                else:
                    print("\nNaive Bayes model")
                    print(f"Accuracy: {round(acc_val,round_dig)} | F1: {round(f1_val,round_dig)} | Recall: {round(recall_val,round_dig)} | Precision: {round(precision_val,round_dig)}")

            model_spec['NB'] = {'cv' : self.cv, 'model' : clf, "y_data" : {"y_test_real": y_test_trans, "y_test_pred" : y_pred_trans, "labels" : lbls}, "accuracy" : acc_val, "f1" : f1_val, "recall" : recall_val, "precision" : precision_val}

"""