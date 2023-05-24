import csv
import itertools
import pandas as pd

# Note: paths to datasets and files are removed
def creating_dataset(root):
    # read in the csv as a dataframe
    prudential_data = pd.read_csv(root)

    original_categorical_featurenames = [
        "Product_Info_1",
        "Product_Info_2",
        "Product_Info_3",
        "Product_Info_5",
        "Product_Info_6",
        "Product_Info_7",
        "Employment_Info_2",
        "Employment_Info_3",
        "Employment_Info_5",
        "InsuredInfo_1",
        "InsuredInfo_2",
        "InsuredInfo_3",
        "InsuredInfo_4",
        "InsuredInfo_5",
        "InsuredInfo_6",
        "InsuredInfo_7",
        "Insurance_History_1",
        "Insurance_History_2",
        "Insurance_History_3",
        "Insurance_History_4",
        "Insurance_History_7",
        "Insurance_History_8",
        "Insurance_History_9",
        "Family_Hist_1",
        "Medical_History_2",
        "Medical_History_3",
        "Medical_History_4",
        "Medical_History_5",
        "Medical_History_6",
        "Medical_History_7",
        "Medical_History_8",
        "Medical_History_9",
        "Medical_History_11",
        "Medical_History_12",
        "Medical_History_13",
        "Medical_History_14",
        "Medical_History_16",
        "Medical_History_17",
        "Medical_History_18",
        "Medical_History_19",
        "Medical_History_20",
        "Medical_History_21",
        "Medical_History_22",
        "Medical_History_23",
        "Medical_History_25",
        "Medical_History_26",
        "Medical_History_27",
        "Medical_History_28",
        "Medical_History_29",
        "Medical_History_30",
        "Medical_History_31",
        "Medical_History_33",
        "Medical_History_34",
        "Medical_History_35",
        "Medical_History_36",
        "Medical_History_37",
        "Medical_History_38",
        "Medical_History_39",
        "Medical_History_40",
        "Medical_History_41",
        "Medical_Keyword_1",
        "Medical_Keyword_2",
        "Medical_Keyword_3",
        "Medical_Keyword_4",
        "Medical_Keyword_5",
        "Medical_Keyword_6",
        "Medical_Keyword_7",
        "Medical_Keyword_8",
        "Medical_Keyword_9",
        "Medical_Keyword_10",
        "Medical_Keyword_11",
        "Medical_Keyword_12",
        "Medical_Keyword_13",
        "Medical_Keyword_14",
        "Medical_Keyword_15",
        "Medical_Keyword_16",
        "Medical_Keyword_17",
        "Medical_Keyword_18",
        "Medical_Keyword_19",
        "Medical_Keyword_20",
        "Medical_Keyword_21",
        "Medical_Keyword_22",
        "Medical_Keyword_23",
        "Medical_Keyword_24",
        "Medical_Keyword_25",
        "Medical_Keyword_26",
        "Medical_Keyword_27",
        "Medical_Keyword_28",
        "Medical_Keyword_29",
        "Medical_Keyword_30",
        "Medical_Keyword_31",
        "Medical_Keyword_32",
        "Medical_Keyword_33",
        "Medical_Keyword_34",
        "Medical_Keyword_35",
        "Medical_Keyword_36",
        "Medical_Keyword_37",
        "Medical_Keyword_38",
        "Medical_Keyword_39",
        "Medical_Keyword_40",
        "Medical_Keyword_41",
        "Medical_Keyword_42",
        "Medical_Keyword_43",
        "Medical_Keyword_44",
        "Medical_Keyword_45",
        "Medical_Keyword_46",
        "Medical_Keyword_47",
        "Medical_Keyword_48",
    ]

    categorical_featurenames = []
    num_featurenames = [
        "Product_Info_4",
        "Ins_Age",
        "Ht",
        "Wt",
        "BMI",
        "Employment_Info_1",
        "Employment_Info_4",
        "Employment_Info_6",
        "Insurance_History_5",
        "Family_Hist_2",
        "Family_Hist_3",
        "Family_Hist_4",
        "Family_Hist_5",
    ]
    disc_featurenames = [
        "Medical_History_1",
        "Medical_History_10",
        "Medical_History_15",
        "Medical_History_24",
        "Medical_History_32",
    ]

    for category in original_categorical_featurenames:
        encoded = pd.get_dummies(prudential_data[category], dtype=float)
        encoded = encoded.add_suffix("_" + category)
        prudential_data = pd.concat([prudential_data, encoded], axis=1).drop(
            category, axis=1
        )
        new_features = list(encoded.columns.values)
        categorical_featurenames.extend(new_features)
    for feature in num_featurenames:
        prudential_data[feature] = (
            prudential_data[feature] - (prudential_data[feature]).mean()
        ) / prudential_data[feature].std()

    # fill in with desired csv name
    prudential_data.to_csv()

    
    # Much of the logic and code in this section is from "Conditionally Risk-Averse Contextual Bandits" [Farsang et al 2022]
    # Paper Link: https://arxiv.org/pdf/2210.13573.pdf
    
    # fill in with the csv that prudential_data was just used to create
    with open("", encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile, delimiter=",")
        rows_header = next(rows, None)
        rows_iter = itertools.islice(rows, 0, None)
        headers_dict = {k: v for v, k in enumerate(rows_header)}
        reverse_headers_dict = {v: k for v, k in enumerate(rows_header)}
        categorical_indices = [headers_dict[name] for name in categorical_featurenames]
        numeric_indices = [headers_dict[name] for name in num_featurenames]
        discrete_indices = [headers_dict[name] for name in disc_featurenames]

        context = {}
        for row in rows_iter:
            for i in categorical_indices:
                if row[i] != "":
                    if "cat_" + str(i) in context:
                        context["cat_" + str(i)].append(row[i])
                        context["cat_" + str(i) + "_NA"].append(0)
                    else:
                        context["cat_" + str(i)] = [row[i]]
                        context["cat_" + str(i) + "_NA"] = [0]

            for i in numeric_indices:
                curr_mean = prudential_data[reverse_headers_dict[i]].mean()
                if row[i] != "":
                    normalized = (float(row[i]) - curr_mean) / (
                        prudential_data[reverse_headers_dict[i]].std()
                    )
                    # filled in default values with 0
                    if "num_" + str(i) in context:
                        context["num_" + str(i)].append(float(normalized))
                        context["num_" + str(i) + "_NA"].append(0)
                    else:
                        context["num_" + str(i)] = [float(normalized)]
                        context["num_" + str(i) + "_NA"] = [0]
                else:
                    if "num_" + str(i) in context:
                        context["num_" + str(i) + "_NA"].append(1)
                        context["num_" + str(i)].append(0.0)
                    else:
                        context["num_" + str(i) + "_NA"] = [1]
                        context["num_" + str(i)] = [0.0]

            for i in discrete_indices:
                curr_mean = prudential_data[reverse_headers_dict[i]].mean()
                if row[i] != "":
                    normalized = (float(row[i]) - curr_mean) / (
                        prudential_data[reverse_headers_dict[i]].std()
                    )
                    if "disc_" + str(i) in context:
                        context["disc_" + str(i)].append((normalized))
                        context["disc_" + str(i) + "_NA"].append(0)
                    else:
                        context["disc_" + str(i)] = []
                        context["disc_" + str(i)].append((normalized))
                        context["disc_" + str(i) + "_NA"] = [0]
                else:
                    if "disc_" + str(i) in context:
                        context["disc_" + str(i) + "_NA"].append(1)
                        context["disc_" + str(i)].append(0.0)
                    else:
                        context["disc_" + str(i) + "_NA"] = [1]
                        context["disc_" + str(i)] = [0.0]

        context_df = pd.DataFrame.from_dict(context)

        # fill in with desired name of csv
        context_df.to_csv()


# call creating_dataset function with path to downloaded Prudential dataset csv from Kaggle
# creating_dataset()
