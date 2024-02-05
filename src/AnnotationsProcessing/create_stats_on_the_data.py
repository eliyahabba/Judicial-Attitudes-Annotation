import statsmodels.stats.inter_rater as inter_rater
from krippendorff import alpha
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

from utils import read_annotated_data, get_most_label

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Data')

DATA_STATS_FOLDER = os.path.join(DATA_FOLDER, '../Data/data_stats')


def print_stats(merged_df):
    labels_count = merged_df.groupby('label').count()['sentence_id']

    # pretty print the labels count
    print('labels count:')
    for label, count in labels_count.items():
        print(f'{label}: {count}')


def create_confusion_matrix(data, taggers):
    confusion_matrix = pd.DataFrame(columns=taggers, index=taggers)
    for i in range(len(taggers)):
        for j in range(len(taggers)):
            if i == j:
                confusion_matrix.at[taggers[i], taggers[j]] = 1
            else:
                # Create a temporary dataframe to store the comparison results
                temp_df = data[[taggers[i], taggers[j]]]
                # Count the number of agreements between the two taggers
                # check intersection between each row
                temp_df['aggreamnt_at_least_of_1'] = temp_df.apply(
                    lambda x: len(set(x[taggers[i]]).intersection(set(x[taggers[j]]))), axis=1)
                temp_df['aggreamnt_at_least_of_2'] = temp_df.apply(lambda x: 1 if
                len(set(x[taggers[i]]).intersection(set(x[taggers[j]]))) else 0, axis=1)

                agreement = temp_df['aggreamnt_at_least_of_1'].sum()
                # Calculate the agreement percentage
                agreement_percentage = agreement / len(temp_df)
                confusion_matrix.at[taggers[i], taggers[j]] = agreement_percentage
    confusion_matrix = confusion_matrix.astype(float)
    confusion_matrix.columns.name = None
    confusion_matrix.index.name = None

    # caluclate the avg agreement for each tagger (but not for the diagonal)
    avg_agreement = np.sum(np.triu(confusion_matrix, k=1)) / len(np.nonzero(np.triu(confusion_matrix, k=1))[0])
    # caluclate the avg agreement for each row:
    avg_agreement_per_row = (np.sum(confusion_matrix, axis=1)-1) / (len(confusion_matrix)-1)
    assert np.isclose(avg_agreement, avg_agreement_per_row.mean()), 'avg agreement is not equal to the avg agreement per row'
    # replace column with list of TAGGER1 and TAGGER2 ... TAGGERn
    confusion_matrix.rename(columns={confusion_matrix.columns[i]: f"T{i + 1}"
                                     for i in range(len(confusion_matrix.columns))}, inplace=True)
    # rename index with list of TAGGER1 and TAGGER2 ... TAGGERn
    confusion_matrix.rename(index={confusion_matrix.index[i]: f"T{i + 1}"
                                   for i in range(len(confusion_matrix.index))}, inplace=True)
    confusion_matrix['AVERAGE'] = avg_agreement_per_row.values

    confusion_matrix.to_csv(os.path.join(DATA_STATS_FOLDER, '../confusion_matrix.csv'))
    confusion_matrix.to_csv(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix_unnamed.csv'))
    # create a heatmap with seaborn
    # plt.figure(figsize=(18, 18))
    return confusion_matrix

def plot_one_heatmep(confusion_matrix):
    #rename column 'accuracy on level 0' to 'accuracy'
    confusion_matrix.rename(columns={'accuracy on level 0': 'Accuracy: \n High Level',
                            'accuracy on level 1': 'Accuracy: \n Granular Level',
                             # 'distance from gold - level 0' : 'Distance: \n level 0'
                                     },
                            inplace=True)
    confusion_matrix.drop(columns=['distance from gold - level 0'], inplace=True)

    fig, ax1 = plt.subplots(figsize=(3., 2.2))
    # add title to the main figure, tight_lay
    # out to avoid overlapping
    # fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold'
    fig.tight_layout()
    # fig.suptitle('Confusion Matrix Heatmap: \n Agreement Between Taggers', fontsize=38, y=0.95)
    # # create a title for each subplot
    # ax1.set_title('Agreement On Level 0', fontsize=28)
    # ax2.set_title('Agreement On Level 1', fontsize=28)
    # fig.suptitle('Agreement Between Taggers',y=0.95 , fontsize=12, fontweight='bold')
    # create a title for each subplot
    # ax1.set_title('Agreement On Level 0', fontsize=10)

    # add another row for the average agreement on the table
    confusion_matrix.loc['AVERAGE'] = confusion_matrix.mean()
    confusion_matrix.at["AVERAGE", 'tagger'] = "AVERAGE"
    # add another column for the average agreement on the table
    # confusion_matrix['AVERAGE'] = confusion_matrix.mean(axis=1)

    # # sns.set(font_scale=2)
    g= sns.heatmap(confusion_matrix.set_index('tagger'), annot=True, fmt='.2f', annot_kws = {"fontsize": 10},
                   # cbar_kws= dict(use_gridspec=False,location="left"),
                   # cbar_kws={"shrink": 0.5},
                   cbar=False,
                   cmap='Blues',
                   linewidths=2,
                   linecolor='black',
                   # square=True,
                   ax=ax1)


    # g= sns.heatmap(confusion_matrix.set_index('tagger'), ax=ax1)
    import matplotlib.font_manager as fm

    # The last label is the average agreement, should be in bold in the table and with no rotation
    # Rotate all y-axis tick labels except the last one
    g.set_yticklabels(g.get_yticklabels(), rotation=90, fontsize=7)

    # Get the last y-axis tick label (representing average agreement)
    last_label = g.get_yticklabels()[-1]

    # Set the last label to be bold and without rotation
    last_label.set_fontweight('bold')
    last_label.set_rotation(0)

    # Set the font size for the last label
    last_label.set_fontsize(7)

    # Update the modified label in the y-axis tick labels
    g.set_yticklabels(g.get_yticklabels())

    # Use a suitable font for bold text (change 'Arial' to your preferred font)
    bold_font = fm.FontProperties(weight='bold')
    last_label.set_font_properties(bold_font)

    g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize=8.5)
    # put x label on top
    g.xaxis.tick_top()
    ax1.axvline(x=confusion_matrix.shape[1] - 1, color='black', linewidth=4)

    # add xlabel and ylabel. Xlabel is on top of the heatmap

    # ax1.set_xlabel("Metrics", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Taggers", fontsize=9, fontweight='bold')
    ax1.xaxis.set_label_position('top')

    fig.tight_layout()
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix.pgf'))
    # save the heatmap
    plt.savefig('confusion_matrix1.png', dpi=600)

    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix1.png'), dpi=600)
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix2.png'), dpi=1200)
    plt.close()

def plot_heatmep(confusion_matrix, confusion_matrix2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4))
    # add title to the main figure, tight_layout to avoid overlapping
    # fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold'
    fig.tight_layout()
    # fig.suptitle('Confusion Matrix Heatmap: \n Agreement Between Taggers', fontsize=38, y=0.95)
    # # create a title for each subplot
    # ax1.set_title('Agreement On Level 0', fontsize=28)
    # ax2.set_title('Agreement On Level 1', fontsize=28)
    fig.suptitle('Confusion Matrix Heatmap: \n Agreement Between Taggers',y=0.95 , fontsize=12, fontweight='bold')
    # create a title for each subplot
    ax1.set_title('Agreement On Level 0', fontsize=10)
    ax2.set_title('Agreement On Level 1', fontsize=10)


    # sns.set(font_scale=2)
    g= sns.heatmap(confusion_matrix, annot=True, fmt='.2f', annot_kws = {"fontsize": 12},
                   # cbar_kws= dict(use_gridspec=False,location="left"),
                   # cbar_kws={"shrink": 0.5},
                   cbar=False,
                   cmap='Blues', linewidths=1, linecolor='black', square=True, ax=ax1)
    g.set_yticklabels(g.get_yticklabels(), rotation = 90, fontsize=8)
    g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize=8)
    ax1.axvline(x=confusion_matrix.shape[1] - 1, color='black', linewidth=4)

    g2 = sns.heatmap(confusion_matrix2, annot=True, fmt='.2f', annot_kws = {"fontsize": 12},
                     # cbar_kws= dict(use_gridspec=False,location="left"),
                     # cbar_kws={"shrink": 0.5},
                     cbar=False,
                     cmap='Blues', linewidths=1, linecolor='black', square=True, ax=ax2)
    g2.set_yticklabels(g2.get_yticklabels(), rotation = 90, fontsize=8)
    g2.set_xticklabels(g2.get_xticklabels(), rotation = 0, fontsize=8)
    # ax1_2 = ax1.twinx()
    # Show xticklabels on the top with ax1_2
    # ax1_2.set_xticks(np.arange(confusion_matrix.shape[1]))
    # ax1_2.set_xticklabels(confusion_matrix.columns, rotation=0, fontsize=8)
    # ax1_2.tick_params(axis='x', length=0)
    # ax1_2.tick_params(axis='x', top=True, labeltop=True)

    g2.set_xticklabels(g2.get_xticklabels(), rotation = 0, fontsize=8)
    ax2.axvline(x=confusion_matrix2.shape[1] - 1, color='black', linewidth=4)
    fig.tight_layout()
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix1.pgf'))
    # save the heatmap
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix1.png'), dpi=600)
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix2.png'), dpi=1200)
    plt.close()

def plot_heatmep2(confusion_matrix, confusion_matrix2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # add title to the main figure, tight_layout to avoid overlapping
    # fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold'
    fig.tight_layout()
    # fig.suptitle('Confusion Matrix Heatmap: \n Agreement Between Taggers', fontsize=38, y=0.95)
    # # create a title for each subplot
    # ax1.set_title('Agreement On Level 0', fontsize=28)
    # ax2.set_title('Agreement On Level 1', fontsize=28)
    fig.suptitle('Confusion Matrix Heatmap: \n Agreement Between Taggers', fontsize=38, y=0.95)
    # create a title for each subplot
    ax1.set_title('Agreement On Level 0', fontsize=28)
    ax2.set_title('Agreement On Level 1', fontsize=28)


    sns.set(font_scale=2)
    g= sns.heatmap(confusion_matrix, annot=True, fmt='.2f', annot_kws = {"fontsize": 44},
                   # cbar_kws= dict(use_gridspec=False,location="left"),
                   cbar_kws={"shrink": 0.5},
                   # cbar=False,
                   cmap='Blues', linewidths=1, linecolor='black', square=True, ax=ax1)
    g.set_yticklabels(g.get_yticklabels(), rotation = 90, fontsize = 23)
    g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 23)
    ax1.axvline(x=confusion_matrix.shape[1] - 1, color='black', linewidth=20)

    g2 = sns.heatmap(confusion_matrix2, annot=True, fmt='.2f', annot_kws = {"fontsize": 44},
                     # cbar_kws= dict(use_gridspec=False,location="left"),
                     cbar_kws={"shrink": 0.5},
                     cmap='Blues', linewidths=1, linecolor='black', square=True, ax=ax2)
    g2.set_yticklabels(g2.get_yticklabels(), rotation = 90, fontsize = 23)
    g2.set_xticklabels(g2.get_xticklabels(), rotation = 0, fontsize = 23)
    ax2.axvline(x=confusion_matrix2.shape[1] - 1, color='black', linewidth=20)
    # save the heatmap
    # fig.tight_layout()
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix1.png'), dpi=600)
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'confusion_matrix2.png'), dpi=1200)
    plt.close()

def get_df_pivot_for_taggers(df, label_column='parent_category_label'):
    # Assume your DataFrame is called 'df'
    # drop all the rows that have nan in the label column
    if pd.isna(df[label_column]).any():
        assert pd.isna(df[label_column]).all(), 'there are some nan values in the label column'

    df['labels'] = df[[label_column, label_column + '2']].apply(lambda x: list(x.dropna()), axis=1)
    # group by sentence_id and tagger, and creat a list of values from label column and labal2 column
    # merge all the values from columns label and label2 to one list

    # create a new column with the merged list
    df_pivot = df.pivot(index='sentence_id', columns='tagger', values='labels')
    return df_pivot


def create_agreement_df(df_pivot):
    agreement_number = df_pivot.apply(lambda x: (
        Counter(list(itertools.chain(*x))).most_common()[0]),
                                      axis=1)
    # split the tuple to two columns
    agreement_number = pd.DataFrame(agreement_number.tolist(), index=agreement_number.index)
    agreement_number.columns = ['label', 'agreement_number']
    agreement_number['agreement_percentage'] = agreement_number['agreement_number'] / len(df_pivot.columns)

    # agreement_label = df_pivot.mode(axis=1)
    # trucate the agreement to 2 digits after the dot
    agreement_number['agreement_percentage'] = agreement_number['agreement_percentage'].apply(lambda x: round(x, 2))
    # concat the agreement number and the agreement label. Give a name to the columns
    agreement_number.to_csv(
        os.path.join(DATA_STATS_FOLDER, f'agreement.csv'),
        encoding='utf-8-sig')
    # save the agreement to csv file
    # convert agreement_percentage_counter to percentage from 0-1 scale by multiplying by 100
    agreement_percentage_counter = agreement_number['agreement_percentage'] * 100
    agreement_percentage_counter = Counter(agreement_percentage_counter)
    # create df from the counter, with columns 'agreement_percentage' and 'count of sentences'
    agreement_percentage_counter = pd.DataFrame.from_dict(agreement_percentage_counter, orient='index',
                                                          columns=['count of sentences'])
    # add index column
    agreement_percentage_counter['agreement_percentage'] = agreement_percentage_counter.index
    agreement_percentage_counter = agreement_percentage_counter.reset_index(drop=True)
    # sort the df by agreement_percentage
    agreement_percentage_counter = agreement_percentage_counter.sort_values(by='agreement_percentage')
    # convert percentage to string and add % sign
    agreement_percentage_counter['agreement_percentage'] = agreement_percentage_counter['agreement_percentage'].apply(
        lambda x: str(x) + '%')
    # replace column order
    agreement_percentage_counter = agreement_percentage_counter[['agreement_percentage', 'count of sentences']]
    # save the agreement percentage counter to csv file
    # save the agreement_percentage_counter to csv file
    agreement_percentage_counter.to_csv(
        os.path.join(DATA_STATS_FOLDER, f'agreement_percentage_counter.csv'),
        encoding='utf-8-sig')
    return agreement_number


def plot_agreement(agreement):
    agreement['label'] = agreement['label'].apply(lambda x: x[::-1] if not x.startswith('not') else x)
    count = agreement.groupby('label').count()
    # sort the labels by the number of sentences
    count = count.sort_values(by='agreement_number', ascending=False)
    labels = count.index

    # plot the number of sentences for each label, with y label as the number of sentences, and x label as the label.
    # use malplotlib to plot the graph and not pandas
    plt.figure(figsize=(4, 8))
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.6)  # <-- Change the 0.02 to work for your plot.
    ax = fig.add_subplot(111)
    ax.bar(count.index, count['agreement_number'])
    ax.set_ylabel('Number of sentences')
    ax.set_xlabel('Label')
    ax.set_title(f'Number of sentences for each label (total number of sentences: {len(agreement)})')
    plt.xticks(rotation=90)
    yticks = [i for i in range(0, max(count['agreement_number']) + 1, 1)]
    plt.yticks(yticks)
    plt.show()


def create_df_pivot_for_agreement(df, label_column, map_from_category_to_label):
    df_pivot = df.pivot(index='sentence_id', columns='tagger', values=label_column)

    map_labels_file = os.path.join(DATA_FOLDER, 'Old_Categories.csv')
    map_labels = pd.read_csv(map_labels_file)
    # create map from category to label (category could apper twice with different labels)
    categories = map_labels[map_from_category_to_label].values
    # remove duplicates with save order
    categories = list(dict.fromkeys(categories))
    map_from_category_ids = {categorie: i for i, categorie in enumerate(categories)}
    df_pivot_ids = df_pivot.applymap(lambda x: map_from_category_ids[x] if pd.notna(x) else x)
    df_pivot_ids = df_pivot_ids.transpose()
    return df_pivot, df_pivot_ids

def create_measure_krippendorff_table_for_column(df, label_column, map_from_category_to_label, name):
    # run measure_krippendorff_agreement function for label_column='label' and label_column='label2'
    # and create df with the results
    result1 = measure_krippendorff_agreement(df, label_column=label_column, map_from_category_to_label=map_from_category_to_label,
                                            replace_zeros_with_nan=False)
    result2 = measure_krippendorff_agreement(df, label_column=label_column, map_from_category_to_label=map_from_category_to_label,
                                                replace_zeros_with_nan=True)
    res_df = pd.DataFrame([result1, result2])
    # add index column with names of params
    res_df['params'] = [name, name +": with out not relevant labels"]
    res_df.set_index('params', inplace=True)
    return res_df

def create_measure_krippendorff_table(df):
    label_columns = ['parent_category_label', 'label']
    map_from_category_to_labels = ['parent_category', 'category']
    names = ['level 0', 'level 1']
    results = []
    for label_column, map_from_category_to_label, name in zip(label_columns, map_from_category_to_labels, names):
        label1_res = create_measure_krippendorff_table_for_column(df, label_column=label_column, map_from_category_to_label=map_from_category_to_label,
                                                                  name=name)
        results.append(label1_res)
    results = pd.concat(results)
    #rename columns
    results.columns = ["Krippendorff's alpha - nominal", "Krippendorff's alpha - ordinal", "fleiss' kappa"]
    # trunc to 2 decimal places
    results = results.round(2)

    results.to_csv(
        os.path.join(DATA_STATS_FOLDER, f'krippendorff_agreement.csv'),
        encoding='utf-8-sig')
    return results


def measure_krippendorff_agreement(df, label_column='label', map_from_category_to_label='category',
                                   replace_zeros_with_nan=False):
    df_pivot_labels, df_pivot = create_df_pivot_for_agreement(df, label_column, map_from_category_to_label)
    df_pivot2_labels, df_pivot2 = create_df_pivot_for_agreement(df, label_column + '2', map_from_category_to_label)
    # take minimum of the two agreement (if the value is nan, take the other value)

    df_pivot2.fillna(df_pivot, inplace=True)
    # combine the two df to one df and take the minimum of the two values
    df_pivot_ids = df_pivot.where(df_pivot < df_pivot2, df_pivot2)
    # convert type to int
    df_pivot_ids = df_pivot_ids.astype(int)

    # replca zeros with nan
    if replace_zeros_with_nan:
        df_pivot_ids = df_pivot_ids.replace(4, np.nan)

    alpha_value = alpha(reliability_data=df_pivot_ids, level_of_measurement="nominal")
    alpha_value_ordinal = alpha(reliability_data=df_pivot_ids, level_of_measurement="ordinal")
    print(f'Krippendorff\'s alpha value for nominal level of measurement: {alpha_value}')
    print(f'Krippendorff\'s alpha value for ordinal level of measurement: {alpha_value_ordinal}')

    # calculate the fleiss_kappa value
    subject_to_cat_counts = inter_rater.aggregate_raters(df_pivot_ids.transpose())
    fleiss_kappa_value = inter_rater.fleiss_kappa(subject_to_cat_counts[0])
    print(f'Fleiss\'s kappa value: {fleiss_kappa_value}')
    return alpha_value, alpha_value_ordinal, fleiss_kappa_value


def get_shared_sentences():
    df = read_annotated_data()
    shared_df = df.groupby('sentence_id').filter(lambda x: len(set(x['tagger'])) > 5)
    shared_df = shared_df[shared_df['status'] == 'annotated']
    shared_df = shared_df.reset_index(drop=True)
    return shared_df


def add_parent_category(df):
    # add parent category column to the df, by reading the parent category from the Categories.csv file
    map_labels_file = os.path.join(DATA_FOLDER, 'Old_Categories.csv')
    map_labels = pd.read_csv(map_labels_file)
    map_labels = map_labels.set_index('category')
    # add not relevant category
    df['parent_category_label'] = df['label'].apply(lambda x: map_labels.loc[x]['parent_category'])
    df['parent_category_label2'] = df['label2'].apply(
        lambda x: map_labels.loc[x]['parent_category'] if not pd.isna(x) else x)
    return df


def get_avg_agg(shared_df):
    map_labels = pd.read_csv(r'C:\Users\t-eliyahabba\PycharmProjects\sentences_annotation_tool\Data\Old_Categories.csv')
    map_labels = map_labels.set_index('category')
    accs1 = []
    accs2 = []
    diffs_type2 = []
    for tagger in shared_df.tagger.unique():
        not_tagger_df = shared_df[shared_df['tagger'] != tagger]
        not_tagger_df = get_most_label(not_tagger_df)
        # take the index and create new column with the df.indrx
        not_tagger_df['sentence_id'] = not_tagger_df.index

        tagger_df = shared_df[shared_df['tagger'] == tagger]
        # not_tagger_df.reset_index(drop=True, inplace=True)
        tagger_df.set_index('sentence_id', inplace=True)
        #sort by index
        tagger_df.sort_index(inplace=True)
        not_tagger_df.sort_index(inplace=True)


        acc_type1 = 0
        acc_type2 = 0
        false_type1 = []
        sum_ = 0
        for i, row in tagger_df.iterrows():
            # take the row idnex from the tagger df and take the row from the not tagger df
            current_val = not_tagger_df.loc[i]
            current_val = pd.DataFrame(current_val).transpose()

            sum_ += 1
            if row.label in current_val.label.values or (not pd.isnull(row.label2)  and row.label2 in current_val.label2.values):
                acc_type1 += 1
                false_type1.append(0)
            else:
                gold_parnet1 = map_labels.loc[current_val['label'].values[0]].parent_category
                gold_parnet2 = None if pd.isnull(current_val['label2'].values[0]) else map_labels.loc[
                    current_val['label2'].values[0]].parent_category

                tagger_parnet1 = map_labels.loc[row['label']].parent_category
                tagger_parnet2 = None if pd.isnull(row['label2']) else map_labels.loc[
                    row['label2']].parent_category
                if tagger_parnet1 in [gold_parnet1, gold_parnet2] or (tagger_parnet2 is not None and tagger_parnet2 in [gold_parnet1, gold_parnet2]):
                    acc_type2 += 1
                    false_type1.append(0)

                else:
                    # continue
                    diff1 = calculate_the_mistake(tagger_parnet1, gold_parnet1)
                    diff2 = calculate_the_mistake(tagger_parnet1, gold_parnet2)
                    diff3 = calculate_the_mistake(tagger_parnet2, gold_parnet1)
                    diff4 = calculate_the_mistake(tagger_parnet2, gold_parnet2)
                    diff = [diff1, diff2, diff3, diff4]
                    # false_type1.append(min(diff))

        # get the accuracy
        print(f'tagger: {tagger}')
        accs1.append(acc_type1 / sum_)
        accs2.append((acc_type1+acc_type2) / sum_)
        # diffs_type2.append(np.mean(false_type1))
        print(f'accuracy type 1: {acc_type1 / sum_}')
        print(f'accuracy type 2: {(acc_type1+ acc_type2 )/ sum_}')
        # print(f'false type 1: {np.mean(false_type1)}')

    # print mean and std of the accuracy
    print(f'mean accuracy type 1: {np.mean(accs1)}')
    print(f'std accuracy type 1: {np.std(accs1)}')
    print(f'mean accuracy type 2: {np.mean(accs2)}')
    print(f'std accuracy type 2: {np.std(accs2)}')

    print(f'mean diffs type 2: {np.mean(diffs_type2)}')
    print(f'std diffs type 2: {np.std(diffs_type2)}')
    # create a df with the accuracy and the tagger name
    df_acc = pd.DataFrame({'tagger':range(1,6), 'accuracy on level 0': accs2, 'accuracy on level 1': accs1,
                           'distance from gold - level 0': accs2})
    df_acc['tagger'] = df_acc['tagger'].apply(lambda x: f'T{int(x)}')
    plot_one_heatmep(df_acc)
    df_acc.to_csv(r'C:\Users\t-eliyahabba\PycharmProjects\sentences_annotation_tool\Data\data_stats\avg_taggers.csv', index=False)

    # create a ploy with table of the accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df_acc.values, colLabels=df_acc.columns, loc='center')
    plt.savefig(r'C:\Users\t-eliyahabba\PycharmProjects\sentences_annotation_tool\Data\data_stats\avg_taggers.png')


    return

def cal_shared_statistics():
    shared_df = get_shared_sentences()
    # drop not relevant sentences
    shared_df = add_parent_category(shared_df)
    get_avg_agg(shared_df)
    df_pivot = get_df_pivot_for_taggers(shared_df, label_column='parent_category_label')
    confusion_matrix = create_confusion_matrix(df_pivot, df_pivot.columns)

    df_pivot = get_df_pivot_for_taggers(shared_df, label_column='label')
    confusion_matrix2 = create_confusion_matrix(df_pivot, df_pivot.columns)
    plot_heatmep(confusion_matrix, confusion_matrix2)
    create_agreement_df(df_pivot)

    # create table with the agreement for each label column
    krippendorff_table = create_measure_krippendorff_table(shared_df)

    # plot_agreement(agreement_df)

def calculate_the_mistake(model_parent, gold_parnet):
    if pd.isnull(model_parent) or pd.isnull(gold_parnet):
        return 11
    map_labels2 = pd.read_csv(r'C:\Users\t-eliyahabba\PycharmProjects\sentences_annotation_tool\Data\Old_Categories.csv')
    #check the difference between the model and the real parent in index from the map_labels
    # set the index to parnet_category instead of category
    map_labels2.set_index('parent_category', inplace=True)
    parent_id = list(set(map_labels2.loc[model_parent, 'parent_category_id'].values)) if isinstance(map_labels2.loc[model_parent, 'parent_category_id'], np.ndarray) else [map_labels2.loc[model_parent, 'parent_category_id']]
    gold_parent_id = list(set(map_labels2.loc[gold_parnet, 'parent_category_id'].values)) if isinstance(map_labels2.loc[gold_parnet, 'parent_category_id'], np.ndarray) else [map_labels2.loc[gold_parnet, 'parent_category_id']]
    # take the minimum difference between the two
    diff = [abs(x - y) for x in parent_id for y in gold_parent_id]
    return min(diff)


if __name__ == "__main__":
    cal_shared_statistics()
