
""" ZHR様。
データ：
https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction
https://www.kaggle.com/datasets/sneharshinde/ltfs-av-data

図の設計：
 - UMAP/tSNE
 - Boxplot
 - Heatmap
 -  - Correlation matrix
 -  - Hierarchical clustering

Use: 
conda activate Fin1

$ python -V
Python 3.9.12

Lasso回帰：数十個の特徴量の中から、片手で数えられるくらいに絞り込む。

LTV: Loan to Value, 借入比率。


"""


import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import pickle

import sklearn
from sklearn.linear_model import Lasso, lasso_path, LassoCV
from sklearn import manifold
from sklearn.decomposition import PCA

import umap


""" File I/O-related functions.
"""
def get_table_file_path(key = "train_Root"):
    lex = {
            "train_Root": "./datasets/l_and_t_vehicle_loan/train.csv", 
            "train_Total": "./datasets/train_Total.csv", 
            "train_Category": "./datasets/train_Category.csv", 
            "stat_MannWhitneyU": "./stat_MannWhitneyU.csv", 
            "stat_FishersExactOnOverdueThreshold": "./stat_FishersExactOnOverdueThreshold.csv", 

            }
    return lex[key]


def read_table_by_key(key = "train_Root"):
    return pd.read_table(get_table_file_path(key), sep = ",")


def save_by_pickle(thing, out_file_path = "./out.pickle"):
    out_file = open(out_file_path, "wb")
    pickle.dump(thing, out_file)
    out_file.close()
    return


def load_by_pickle(pickle_file_path = "./out.pickle"):
    pickle_file = open(pickle_file_path, "rb")
    thing = pickle.load(pickle_file)
    return thing


def get_list_file_path(key = "list_Employee"):
    lex = {
            "list_Employee": "./datasets/list_Employee.pickle", 
            "list_CountsLtv": "./datasets/list_CountsLtv.pickle", 
            "list_PercentileToBankruptRate": "./datasets/list_PercentileToBankruptRate.pickle", 
            "list_BankruptRateOverInversePercentile": "./datasets/list_BankruptRateOverInversePercentile.pickle", 
            }
    return lex[key]


def load_list_by_key(key = "list_Employee"):
    return load_by_pickle(get_list_file_path(key))


def exclude_train_frame_outliers(train_frame, debug = False):
    if debug:
        idx = 71593
        print(train_frame.loc[idx, "UniqueID"])
        print(train_frame.iloc[idx, :])
        print(train_frame.iloc[(idx - 3):(idx + 3), :].loc[:, 
                ["disbursed_amount", "ltv", 
                        "PRI.SANCTIONED.AMOUNT", "PRI.DISBURSED.AMOUNT"]])
    
    submit_frame = train_frame.query("UniqueID not in [598208, 629503, 556040, 488582, 440173, 585144, 489321]")
    return submit_frame


def exclude_train_frame_non_matrix_columns(train_frame, debug = False):
    if debug:
        print(train_frame.head())
    columns = [
            e for e in train_frame.columns 
            if e not in ['Unnamed: 0', 'UniqueID', 'Employee_code_ID', 'loan_default']]
    submit_frame = train_frame.loc[:, columns]
    return submit_frame


def make_train_frame_total(debug = False):
    train_frame = read_table_by_key("train_Root")
    # Process for sanitizing.
    # This time, focus on initial analysis.
    # Columns needing conversion: [
    #       "Date.of.Birth", "DisbursalDate", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"
    #       ]
    # "UniqueID" is ID for customers.
    # "Date.of.Birth", "Employee_code_ID", are excluded.
    columns_select = [
            "UniqueID", 
            "disbursed_amount", "asset_cost", "ltv", "PRI.NO.OF.ACCTS", 
            "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", 
            "PRI.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", 
            "NO.OF_INQUIRIES", "loan_default"
            ]
    transfer_frame = train_frame.loc[:, columns_select]
    transfer_frame = exclude_train_frame_outliers(transfer_frame)
    submit_frame = transfer_frame

    if debug:
        print(train_frame.head())
        print(len(train_frame["Date.of.Birth"].unique()))
    
    submit_frame.to_csv(get_table_file_path("train_Total"))


def make_train_frame_category(debug = False):
    train_frame = read_table_by_key("train_Root")
    # Select only categorical values: [
    #       "branch_id", "supplier_id", "manufacturer_id", "State_ID", 
    #       "Employment.Type", "PERFORM_CNS.SCORE.DESCRIPTION"]
    columns_select = [
            "UniqueID", "Employee_code_ID", 
            "Employment.Type", "loan_default"
            ]
    submit_frame = train_frame.loc[:, columns_select]
    if debug:
        print(train_frame.head())
    submit_frame.to_csv(get_table_file_path("train_Category"))


def make_train_list_employee(verbose = True, debug = False):
    train_frame = read_table_by_key("train_Root")
    people = train_frame["Employee_code_ID"].unique()

    submit_list = list()
    length = len(people)
    for i in range(length):
        person = people[i]
        if verbose and i % 50 == 0:
            print("i = {} / {} ...".format(i, length))
        person_frame = train_frame.query("Employee_code_ID == {}".format(person))
        submit_list.append(person_frame)
    
    if debug:
        print("# People:", len(people))
        print(train_frame.head())
        print(person_frame.head())
        print(person_frame.shape)
    
    if verbose: print("Saving...")
    save_by_pickle(submit_list, out_file_path = get_list_file_path("list_Employee"))
    return


""" Statistics-related functions.
"""
def test_Fishers_exact_test(debug = True):
    data = pd.DataFrame({
            "Doctor":["Tarou","Tarou","Hanako","Hanako"], 
            "Operation":["Endoscope","Dissection","Endoscope","Dissection"], 
            "Times":[60,5,20,40]})
    table = pd.pivot_table(data, values = "Times", index = "Doctor", columns = "Operation", aggfunc = "sum")
    if debug:
        print(data)
        print(table)
        print(stats.fisher_exact(table)) #Fisher exact test
        print(stats.chi2_contingency(table)) #カイ2乗検定（イエーツ補正あり）
    return


def get_report_fishers_exact_test_with_threshold(
        bankrupts, solvents, 
        alt = "two-sided",  # {‘two-sided’, ‘less’, ‘greater’},
        theta = None, debug = False):
    if theta is None:
        theta = pd.concat([bankrupts, solvents]).quantile()
    
    ge_b = sum(bankrupts >= theta)
    lt_b = sum(bankrupts < theta)
    ge_s = sum(solvents >= theta)
    lt_s = sum(solvents < theta)

    data = pd.DataFrame({
            "Loan_Default":["Bankrupt","Bankrupt","Solvent","Solvent"], 
            "Threshold":["Greater_Equal","Less_Than","Greater_Equal","Less_Than"], 
            "Count":[ge_b, lt_b, ge_s, lt_s]})
    #data = pd.DataFrame({"術者":["太郎","太郎","花子","花子"], "手術方法":["内視鏡","開腹","内視鏡","開腹"], "回数":[60,5,20,40]})
    table = pd.pivot_table(data, values = "Count", index = "Threshold", columns = "Loan_Default", aggfunc = "sum")
    #table = pd.pivot_table(data, values = "回数", index = "術者", columns = "手術方法", aggfunc = "sum")
    report = stats.fisher_exact(table, alternative = alt)

    if debug:
        pvalue = report[1]
        print(data)
        print(table)
        print(pvalue)

    return report


def get_report_wilcoxon_rank_sum(
        bankrupt_frame, solvent_frame, 
        column = "ltv", 
        debug = False
        ):
    report = stats.ranksums(
            bankrupt_frame[column], 
            solvent_frame[column], 
            alternative = 'two-sided')
    if debug: 
        print(report)
        print(type(report))
        print("Pvalue:", report[1])
        print(bankrupt_frame[column].loc[0:30])
    return report


def get_report_mann_whitney_u(
        bankrupt_frame, solvent_frame, 
        column = "ltv", 
        debug = False
        ):
    report = stats.mannwhitneyu(
            bankrupt_frame[column], 
            solvent_frame[column], 
            alternative = 'two-sided', 
            method = "asymptotic"
            )
    if debug: 
        print(report)
        print(type(report))
        print("Pvalue:", report[1])
        print(bankrupt_frame[column].loc[0:30])
    return report


""" Pre-development bare exploratory functions.
"""
def explore_l_and_t_table_heads(train_frame):
    columns = train_frame.columns
    for column in columns:
        print(column)
        print(train_frame[column].loc[0:20])
    return
def drive_explore_l_and_t_table_heads():
    train_frame = read_table_by_key("train_Root")
    explore_l_and_t_table_heads(train_frame)


def explore_l_and_t_table_categorical(
        train_frame, 
        column = "Employment.Type"
        ):
    print("#", column, ":")
    print(train_frame[column].value_counts(dropna = False))
    print(set(train_frame[column]))
    return


def explore_l_and_t_table_categoricals(train_frame):
    # Select only categorical values: ["branch_id", "supplier_id", "manufacturer_id", "State_ID", "Employment.Type", "PERFORM_CNS.SCORE.DESCRIPTION"]
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "branch_id"
            )
    """ Supressed since it is too long.
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "supplier_id"
            )
    """
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "manufacturer_id"
            )
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "State_ID"
            )
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "Employment.Type"
            )
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "PERFORM_CNS.SCORE.DESCRIPTION"
            )
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "NEW.ACCTS.IN.LAST.SIX.MONTHS"
            )
    explore_l_and_t_table_categorical(
            train_frame, 
            column = "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"
            )
    return
def drive_explore_l_and_t_table_categoricals():
    train_frame = read_table_by_key("train_Root")
    explore_l_and_t_table_categoricals(train_frame)


def explore_l_and_t_table_column(column = "UniqueID"):
    train_frame = read_table_by_key("train_Root")
    ids = train_frame[column]
    print(len(ids))
    print(len(set(ids)))
    return


def explore_l_and_t_table_unique_ids():
    explore_l_and_t_table_column("UniqueID")
def explore_l_and_t_table_employee_ids():
    explore_l_and_t_table_column("Employee_code_ID")
    # Thus, it has identical people recorded multiple times.


""" Exploratory analysis.
"""
def explore_playground_1(debug = True):
    desc_frame = pd.read_table("./datasets/l_and_t_vehicle_loan/data_dictionary.csv", sep = ",")
    test_frame = pd.read_table("./datasets/l_and_t_vehicle_loan/test.csv", sep = ",")
    train_frame = read_table_by_key("train_Root")

    if debug:
        print(test_frame.head())
        print(test_frame.columns)
        print(train_frame.head())
        print(train_frame.columns)
        print(desc_frame.head(25))
        print(desc_frame.columns)
        print(desc_frame.iloc[:, 1:3])
        # Loan EMI: equated monthly installment.
        # 月額分割額。
        # Disbursement: 貸出実行額、ローンの実行部分。
        # Loan sanction letter: ローンが承認されたことを通知する書類。
        # Principal Outstanding Amount: ローン残債部分。元本残債。
        # Loan tenure: ローン期間。
        # Loan inquiry: ローンを借りようとする問い合わせ。

        # Candidates: [
        #   "loan_default", "disbursed_amount", "asset_cost", "ltv", 
        #   "Date.of.Birth", "Employment.Type", "DisbursalDate", "State_ID", 
        #   "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.SANCTIONED.AMOUNT", 
        #   "PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", 
        #   "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", 
        #   "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH", "NO.OF_INQUIRIES", 
        #   ]
        # Strings: ["DisbursalDate", "Date.of.Birth", "Employment.Type", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"]
        # Floats: ["ltv"]
        # Integers: ["disbursed_amount", "asset_cost", "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "NO.OF_INQUIRIES"]
        # Booleans: ["loan_default"]

        print(test_frame["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"].head(50))
        print(train_frame["loan_default"].head(50))
        # "loan_default" is only in train_frame .

        #IF
    return


def explore_playground_2(debug = True):
    
    train_frame = read_table_by_key("train_Root")

    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")
    
    report = get_report_wilcoxon_rank_sum(
            bankrupt_frame, solvent_frame, 
            column = "ltv"
            )
    
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(train_frame.shape)
        print(bankrupt_frame.head(20))
        print(bankrupt_frame.shape)
        print(solvent_frame.head(20))
        print(solvent_frame.shape)

    #explore_playground_1()
    explore_l_and_t_table_heads(train_frame)
    #test_Fishers_exact_test()
    #explore_l_and_t_table_categoricals(train_frame)
    #explore_l_and_t_table_unique_ids(train_frame)

    return


def explore_playground_3(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    #['Unnamed: 0', 'UniqueID', 'Employee_code_ID', 'loan_default']
    X_train = exclude_train_frame_non_matrix_columns(train_frame)
    y_train = train_frame["loan_default"]

    model = Lasso(alpha = 0.5, max_iter = int(1.0E+4)).fit(X_train, y_train)

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(X_train.columns)
        print(X_train.head())
        print(y_train.head())
        print(model.score(X_train, y_train))
        print(np.sum(model.coef_ != 0))
    
    return


def explore_playground_4(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    #['Unnamed: 0', 'UniqueID', 'Employee_code_ID', 'loan_default']
    X_train = exclude_train_frame_non_matrix_columns(train_frame)
    y_train = train_frame["loan_default"]

    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps = 5E-3)
    
    # 解パスを描写する
    plt.figure(1)
    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso Paths")
    l1[-1].set_label("Lasso")
    plt.legend()
    plt.axis("tight")
    plt.show()

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(X_train.columns)
        print(X_train.head())
        print(y_train.head())
    
    return


def explore_playground_5(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    #['Unnamed: 0', 'UniqueID', 'Employee_code_ID', 'loan_default']
    X_train = exclude_train_frame_non_matrix_columns(train_frame)
    y_train = train_frame["loan_default"]
    # CVの実行
    lasso = LassoCV(cv=20, random_state=0, max_iter = int(1.0E4)).fit(X_train, y_train)

    # 結果のグラフ化
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="Best alpha")
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.title("MSE for each alpha")
    plt.legend()
    plt.show()

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(X_train.columns)
        print(X_train.head())
        print(y_train.head())
    
    return


def save_table_mann_whitney_u(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    
    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")
    
    columns = [
            'disbursed_amount',
            'asset_cost', 'ltv', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',
            'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            'NO.OF_INQUIRIES']
    
    large_N = len(train_frame)
    records = list()
    for column in columns:
        print(column)
        report = get_report_mann_whitney_u(
                bankrupt_frame, solvent_frame, 
                column = column
                )
        pvalue = report[1]
        print(pvalue)
        score = - np.log10(pvalue)
        record = [column] + list(report) + [score, large_N]
        records.append(record)

        if debug:
            print(record)
    
    submit_frame = pd.DataFrame(
            records, 
            columns = [
                    "column", "statistic", "pvalue", "score", 
                    "large_N"])
    submit_frame.to_csv(get_table_file_path("stat_MannWhitneyU"))

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(bankrupt_frame.head())
        print(solvent_frame.head())
        print(type(report))
        print(train_frame["ltv"].head())
        print(train_frame["disbursed_amount"].head())
        print(train_frame["asset_cost"].head())
    
    return


def plot_barplot_table_mann_whitney_u(show = False, debug = False):
    stat_frame = read_table_by_key(key = "stat_MannWhitneyU")
    # Exclude p == 0.0 cases.
    stat_frame = stat_frame.query("pvalue != 0.0")
    stat_frame = stat_frame.sort_values("score", ascending = True)
    large_N = stat_frame.loc[1, "large_N"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(
            np.arange(len(stat_frame)), 
            stat_frame["score"], 
            tick_label = stat_frame["column"])
    ax.set_xlabel('Significance [- log10(P)]')
    ax.set_ylabel('Variable')
    plt.title("Mann Whitney U test, N = %.1E; ZHR" % (large_N))
    plt.subplots_adjust(left = 0.5)
    # By default, left == 0.125 .
    plt.savefig("./figure_MannWhitneyU.pdf")
    if show: plt.show()

    if debug:
        print(stat_frame.columns)
        print(stat_frame.head())
        scores = stat_frame["score"]
        print(scores)
        #IF
    return


def explore_playground_6(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())


def select_train_frame_columns_negative(train_frame):
    header = train_frame.columns
    columns = [
            column for column in header
            if column not in ['Unnamed: 0', 'UniqueID', 'Employee_code_ID', 'loan_default']
            ]
    return columns


def explore_playground_7(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    columns = select_train_frame_columns_negative(train_frame)

    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")

    for column in columns:
        print(column)
        bankrupts = bankrupt_frame[column]
        solvents = solvent_frame[column]
        report = get_report_fishers_exact_test_with_threshold(
                bankrupts, solvents, 
                alt = "greater", 
                theta = None, debug = False)
        pvalue = report[1]
        print(pvalue)

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(bankrupt_frame.head())
        print(solvent_frame.head())
        print(columns)
        print(train_frame["ltv"].head(30))
    
    return


def explore_playground_8(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    columns = select_train_frame_columns_negative(train_frame)

    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")

    column = "ltv"

    if debug:
        print(len(column))
    
    figure = plt.figure()
    count_row = 6
    axes = [figure.add_subplot(count_row, 1, i) for i in range(1, count_row + 1)]

    #fig, ax = plt.subplots()

    def add_boxplot(ax, column, bankrupt_frame, solvent_frame):
        bp = ax.boxplot(
                (bankrupt_frame[column], solvent_frame[column]), 
                notch = True, vert = False, 
                showfliers = False)
        ax.set_yticklabels(['bankrupt', 'solvent'])
        plt.title(column)
        plt.grid()

    for i in range(count_row):
        add_boxplot(axes[i], columns[i], bankrupt_frame, solvent_frame)
    
    plt.show()

    return


def explore_playground_9(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    columns = select_train_frame_columns_negative(train_frame)

    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")
    
    #column = "disbursed_amount"
    column = "PRI.OVERDUE.ACCTS"
    bankrupts = bankrupt_frame[column]
    solvents = solvent_frame[column]
    values = pd.concat([bankrupts, solvents])

    #percents = [10 * i for i in range(10)]
    #percents = range(80, 100, 1)
    #theta = values.quantile(0.01 * percent)

    for theta in range(1, 20):
        print("Theta({}): {}".format(0, theta))
        report = get_report_fishers_exact_test_with_threshold(
                bankrupts, solvents, 
                alt = "greater", 
                theta = theta, debug = False)
        pvalue = report[1]
        print("P:", pvalue)
        #FOR
    
    return


def explore_playground_10(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    columns = select_train_frame_columns_negative(train_frame)
    mx_train = train_frame.loc[:, columns]

    mapper = manifold.TSNE(random_state = 0, init = "random", learning_rate = 200)
    embedding = mapper.fit_transform(mx_train)
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]

    if debug:
        print(mx_train.head())
        print(embedding_x)
        print(embedding_y)

    return


def explore_playground_11(debug = True):
    people_list = load_list_by_key(key = "list_Employee")

    person_frame = people_list[5]
    count_default = sum(person_frame["loan_default"])
    count_loan = len(person_frame["DisbursalDate"].unique())

    # ["Date.of.Birth", "Employment.Type", "DisbursalDate",]
    if debug:
        print(type(people_list))
        print(len(people_list))
        print(count_default)
        print(person_frame.head())
        print(set(person_frame["DisbursalDate"]))
        print(count_loan)
        print(set(person_frame["Date.of.Birth"]))
        # Thus, Multiple customers can be recorded by a single employee in the organization.


def explore_playground_12(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())


def plot_pca_map_of_train_total(show = False, sanitize = False, debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    if sanitize: train_frame = exclude_train_frame_outliers(train_frame)

    mx_train = exclude_train_frame_non_matrix_columns(train_frame)
    y_train = train_frame["loan_default"]

    pca = PCA(n_components = 2, random_state = 0)
    x_embedded = pca.fit_transform(mx_train)
    
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c = y_train)
    plt.colorbar()

    plt.title("PCA (yellow: bankrupt, purple: solvent, N = %.1E); ZHR" % (len(train_frame)))
    if show:
        plt.show()
    
    plt.savefig("figure_PcaOfCustomers.png")

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(mx_train.head())
        print(x_embedded)
        print(type(x_embedded))

        embed_frame = pd.DataFrame(x_embedded)
        embed_frame.columns = ["x", "y"]
        select_frame = embed_frame.query("x > 0.8e+9")

        print(embed_frame.head())
        print(select_frame)
        print(max(x_embedded[:, 0]), max(x_embedded[:, 1]))
        #IF
    return


def plot_violins_of_train_frame_initial(sanitize = False, show = False, debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    if sanitize: train_frame = exclude_train_frame_outliers(train_frame)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    columns = ["disbursed_amount", "asset_cost"]
    ax.violinplot(train_frame.loc[:, columns], showmedians = True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(columns)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Disbursed Amount or Asset Cost')
    plt.title("[Disbursed Amount] and [Asset Cost] Distributions; ZHR")
    plt.subplots_adjust(left = 0.2)
    # By default, left == 0.125 .
    if show: plt.show()
    plt.savefig("./figure_ViolinDollars.pdf")
    return


def explore_playground_13(sanitize = False, debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    if sanitize: train_frame = exclude_train_frame_outliers(train_frame)
    # "disbursed_amount", "asset_cost", "ltv",

    def melt(data_frame, column, debug = False):
        submit_frame = data_frame[[column]]
        submit_frame.columns = ["value"]
        submit_frame["variable"] = column
        if debug:
            print(submit_frame.head())
        return submit_frame

    loan_frame = melt(train_frame, "disbursed_amount")
    asset_frame = melt(train_frame, "asset_cost")
    show_frame = pd.concat([loan_frame, asset_frame], axis = 0)

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(train_frame.loc[:, ["disbursed_amount", "asset_cost", "ltv"]].describe())
        rates = train_frame["ltv"]
        print(loan_frame.head())
        values = train_frame["disbursed_amount"]
        values = values.sort_values(ascending = False)
        print(values.head(20))
        high_asset_frame = train_frame.query("asset_cost > 4E+5")
        print(high_asset_frame)
        high_disburse_frame = train_frame.query("disbursed_amount > 3E+5")
        print(high_disburse_frame)
        print(sum(abs(train_frame["ltv"] - 100 * train_frame["disbursed_amount"] / train_frame["asset_cost"])))
        # LTV = [loan value] / [asset value]
        
    

    """
    sns.violinplot(
            x='variable', y='value', data = show_frame, hue='variable', dodge=True,
            jitter=True, color='black', palette='Set3', ax=ax)
    """


def save_list_counts_ltv(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    
    def get_counts_of_ltv(train_frame, debug = False):
        ltvs = train_frame["ltv"]
        ltvs_int = ltvs.astype("int")
        counts = ltvs_int.value_counts()
        if debug:
            print(train_frame["ltv"].describe())
            print(ltvs.head())
            print(ltvs_int.head())
            print(counts)
        return counts

    bankrupt_frame = train_frame.query("loan_default == 1")
    counts_total = get_counts_of_ltv(train_frame)
    counts_bankrupt = get_counts_of_ltv(bankrupt_frame)

    def pull_count_by_int_idx(counts, idx = 70, na = 0):
        if idx not in counts.index: atom = na
        else: atom = counts[idx]
        return atom
    
    rates = list()
    for i in range(0, 99):
        total = pull_count_by_int_idx(counts = counts_total, idx = i, na = 0)
        if total <= 0: total = 1
        bankrupt = pull_count_by_int_idx(counts = counts_bankrupt, idx = i, na = 0)
        rate = float(bankrupt) / total
        rates.append(rate)
    
    # Save ltv_counts_total, ltv_counts_bankrupt, and ltv_bankrupt_rates in pickle.
    lex = dict(
            ltv_counts_total = counts_total, 
            ltv_counts_bankrupt = counts_bankrupt, 
            ltv_bankrupt_rates = rates, 
            large_N = len(train_frame))
    save_by_pickle(lex, get_list_file_path("list_CountsLtv"))

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        bankrupts = train_frame["loan_default"]
        print(bankrupts.head())
        print(counts_total)
        print(counts_bankrupt)
        print(type(counts_total))
        print(counts_total.index)
        print(counts_bankrupt[79])
        #print(counts_bankrupt[1])
        print(rates)
        #IF
    return


def plot_barplot_ltv_vs_bankrupt_rate(show = False, debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    count_list = load_by_pickle(get_list_file_path("list_CountsLtv"))
    rates = count_list["ltv_bankrupt_rates"]
    rests = [1.0 - e for e in rates]
    large_N = count_list["large_N"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    left = range(1, len(rates) + 1)
    width = 0.95
    bar_rate = plt.bar(left, rates, width = width, color = "yellow")
    bar_rest = plt.bar(left, rests, width = width, color = "purple", bottom = rates)
    plt.legend((bar_rate[0], bar_rest[0]), ("Bankrupt", "Solvent"))
    #plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ax.set_xlabel('Loan to Asset Value (%)')
    ax.set_ylabel('Default Rate')
    plt.title("Default Rate vs. Loan to Asset Value, N = %.1E; ZHR" % (large_N))
    xmin = - 5.0; xmax = 105;
    plt.hlines([1.0], xmin, xmax, "black", linestyles = 'solid', linewidth = 0.5)
    plt.xlim(xmin, xmax)
    plt.savefig("figure_BarplotLtvVsDefaultRate.pdf")
    if show: plt.show()

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(count_list)


def save_list_of_bankrupt_rate_matrix(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    
    variable_frame = exclude_train_frame_non_matrix_columns(train_frame)
    variable_frame["loan_default"] = train_frame["loan_default"]
    rank_frame = variable_frame.rank(method = "min")
    # The lower the value, the smaller the rank is.
    norm_frame = rank_frame / len(train_frame)
    percentile_frame = np.floor(100 * norm_frame).astype("int")
    
    def convert_percentiles_to_bankrupt_rates(
            percentile_frame, variable_frame, column, 
            debug = False):
        values_df = percentile_frame[[column]]
        values_series = percentile_frame[column]
        values_unique = values_series.unique()
        atoms = values_df.copy()
        # Avoid chained indexing using df.copy() .
        for value in values_unique:
            indices = np.where(values_df == value)[0].tolist()
            if debug:
                print(indices)
                print(type(indices))
            
            #indices = percentile_frame.query("%s == %d" % (column, value)).index
            if len(indices) > 0:
                select_frame = variable_frame.iloc[indices, :]
                bankrupts = select_frame["loan_default"]
                rate = sum(bankrupts) / len(bankrupts)
            else:
                rate = 0.0
            
            atoms.iloc[indices, 0:1] = rate
            if debug:
                print(values_unique)
                print(indices)
                print(select_frame)
                print(rate)
                break
            #FOR
        if debug:
            print(atoms.head(30))
        return atoms
    
    submit_frame = pd.DataFrame()
    for column in percentile_frame.columns:
        print(column)
        atoms = convert_percentiles_to_bankrupt_rates(
                percentile_frame, variable_frame, 
                column = column)
        submit_frame[column] = atoms

    submit_frame = submit_frame.round(6)
    save_by_pickle(submit_frame, out_file_path = get_list_file_path("list_PercentileToBankruptRate"))

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        values = train_frame["asset_cost"]
        print(values.head(30))
        for column in norm_frame.columns:
            print(norm_frame[column].describe())
        print(percentile_frame.head())
        print(train_frame["loan_default"].quantile(78.0 / 100))
        print(train_frame["loan_default"].quantile(79.0 / 100))
        print(atoms.head(30))
        #IF
    return


def explore_playground_14(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())


def save_list_of_matrix_of_bankrupt_rate_over_inverse_percentile(debug = False):
    """ Sorting a matrix in a special way. """
    rate_frame = load_by_pickle(get_list_file_path("list_PercentileToBankruptRate"))

    def sort_frame_one_by_one(data_frame, verbose = True, debug = False):
        submit_frame = pd.DataFrame()
        for column in data_frame.columns:
            if verbose: print(column)
            temp_frame = data_frame[column].sort_values(ascending = False, inplace = False)
            temp_frame = temp_frame.reset_index().drop("index", axis = 1)
            # Drop the "index" column.
            submit_frame[column] = temp_frame[column]
            if debug:
                print(temp_frame.head())
                print(submit_frame.head())
        return submit_frame
    
    sort_frame = sort_frame_one_by_one(rate_frame)
    display_frame = sort_frame.drop("loan_default", axis = 1).T
    save_by_pickle(display_frame, get_list_file_path("list_BankruptRateOverInversePercentile"))
    
    if debug:
        print(rate_frame.columns)
        print(rate_frame.head())
        print(sort_frame.head())
    return


def plot_heatmap_of_bankrupt_rate_over_inverse_percentile(show = False):
    display_frame = load_by_pickle(get_list_file_path("list_BankruptRateOverInversePercentile"))

    figure, ax = plt.subplots()
    sns.heatmap(display_frame, vmin = 0.0, vmax = 0.4, cmap = "seismic")
    ax.set_xlabel('Loan Default Rate Ranks')
    ax.set_ylabel('Variable')
    plt.title("Loan Default Rate over Inverse Percentile; ZHR")
    plt.subplots_adjust(left = 0.5, bottom = 0.2)
    # By default, left == 0.125 .
    xticks = [0, len(display_frame.columns)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["%.1E" % xticks[i] for i in range(len(xticks))])
    if show: plt.show()
    plt.savefig("figure_HeatmapOfBankruptRateOverInversePercentile.png")
    return


def plot_heatmap_peason_correlation_matrix_with_euclidean(show = False, debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    variable_frame = exclude_train_frame_non_matrix_columns(train_frame)
    mx_corr = variable_frame.corr()

    sns.set(font_scale = 1.4)
    sns_plot = sns.clustermap(
            mx_corr, method = 'ward', 
            metric = 'euclidean', cmap = "viridis", 
            linewidth = 0.5
            )
    plt.setp(sns_plot.ax_heatmap.get_yticklabels(), rotation = 0)
    plt.setp(sns_plot.ax_heatmap.get_xticklabels(), rotation = 90)
    sns_plot.fig.suptitle("Pearson Correlation, Euclidean distance; ZHR", fontsize = 20)
    plt.savefig("figure_MatrixOfPearsonCorrelationEuclidean.pdf")
    if show: plt.show()

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(mx_corr)
        print(mx_corr.shape)
        #IF
    return


def explore_playground_15(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    overdues = train_frame["PRI.OVERDUE.ACCTS"]
    
    def get_bankrupt_rate(data_frame):
        if len(data_frame) > 0:
            bankrupts = data_frame["loan_default"]
            rate = sum(bankrupts) / len(bankrupts)
        else: rate = 0.0
        return(rate)
    
    rates = list()
    for i in range(1, 16):
        print("# d > %d" % (i))
        indices = np.where(overdues > i)[0].tolist()
        overdue_frame = train_frame.iloc[indices, :]
        rate = get_bankrupt_rate(overdue_frame)
        rates.append(rate)

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(overdues.describe())
        print(overdue_frame.loc[:, ["PRI.OVERDUE.ACCTS", "loan_default"]].head())
        print(overdue_frame.shape)
        print(rate)
        print(rates)


def explore_playground_16(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())


def save_table_fishers_exact_test_over_overdue_thresholds(debug = False):
    train_frame = read_table_by_key(key = "train_Total")
    
    bankrupt_frame = train_frame.query("loan_default == 1")
    solvent_frame = train_frame.query("loan_default == 0")
    
    column = "PRI.OVERDUE.ACCTS"
    bankrupts = bankrupt_frame[column]
    solvents = solvent_frame[column]

    large_N = len(train_frame)
    alt = "greater"
    records = list()
    for theta in range(1, 20):
        report = get_report_fishers_exact_test_with_threshold(
                bankrupts, solvents, 
                alt = alt, 
                theta = theta, debug = False)
        pvalue = report[1]
        score = - np.log10(pvalue)
        print("Theta: {}, P = {}".format(theta, pvalue))
        record = [theta] + list(report) + [score, large_N, alt]
        records.append(record)
        #FOR
    
    submit_frame = pd.DataFrame(
            records, 
            columns = [
                    "theta", "statistic", "pvalue", "score", 
                    "large_N", "alternate"])
    submit_frame.to_csv(get_table_file_path("stat_FishersExactOnOverdueThreshold"))

    if debug:
        print(train_frame.columns)
        print(train_frame.head())
        print(bankrupt_frame.head())
        print(solvent_frame.head())
        print(type(report))
    
    return


def plot_barplot_table_fishers_exact_test_over_overdue_thresholds(show = False, debug = False):
    stat_frame = read_table_by_key(key = "stat_FishersExactOnOverdueThreshold")
    # Exclude p == 0.0 cases.
    stat_frame = stat_frame.query("pvalue != 0.0")
    stat_frame = stat_frame.sort_values("theta", ascending = False)
    large_N = stat_frame.loc[1, "large_N"]

    length = len(stat_frame)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(
            np.arange(length), 
            stat_frame["score"], 
            tick_label = stat_frame["theta"])
    ax.set_xlabel('Significance [- log10(P)]')
    ax.set_ylabel('# Primary Overdue Accounts')
    plt.title("Fisher's Exact test, one-sided, \nalpha = 0.05, N = %.1E; ZHR" % (large_N))
    plt.subplots_adjust(left = 0.125)
    # By default, left == 0.125 .
    plt.vlines([- np.log10(0.05)], - 2, length + 2, "black", linestyles = 'dashed', linewidth = 0.5)
    plt.ylim(0, length)

    plt.savefig("./figure_BarplotOfFishersExactOnOverdueThreshold.pdf")
    if show: plt.show()

    if debug:
        print(stat_frame.columns)
        print(stat_frame.head())


def explore_playground_17(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())


def explore_playground_18(debug = True):
    train_frame = read_table_by_key(key = "train_Total")
    
    if debug:
        print(train_frame.columns)
        print(train_frame.head())




if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc >= 2:
        if argc == 2:
            func = argv[1]
            exec(func + "()")
            print("### Executed function:", func)
        #IF
    #MAIN
#EOF