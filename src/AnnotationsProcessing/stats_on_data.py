import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Data')
from utils import read_annotated_data

from create_stats_on_the_data import DATA_STATS_FOLDER, get_shared_sentences

# in this file we calulate the stats on the data and save them to a json file
dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Data')


def calculate_stats_on_results(result_df):
    # count the distribution of the labels
    labels = result_df['label'].value_counts()
    labels = labels.to_frame()
    labels = labels.reset_index()
    labels = labels.rename(columns={'index': 'label', 'label': 'count'})
    labels['percentage'] = labels['count'] / labels['count'].sum()
    labels['percentage'] = labels['percentage'].apply(lambda x: "{:.2%}".format(x))
    labels = labels.sort_values(by=['count'], ascending=False)
    labels = labels.reset_index(drop=True)
    # display the distribution of the labels
    labels.plot.bar(x='label', y='count', rot=0)
    plt.title('Distribution of the labels')
    plt.savefig(os.path.join(DATA_STATS_FOLDER, 'labels_distribution.png'))

    plt.figure(figsize=(4, 8))
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.6)  # <-- Change the 0.02 to work for your plot.
    ax = fig.add_subplot(111)
    labels.plot.bar(x='label', y='count', rot=0)
    ax.bar(labels.label, labels.count)
    ax.set_ylabel('Number of sentences')
    ax.set_xlabel('Label')
    ax.set_title(f'Number of sentences for each label (total number of sentences: {len(labels)})')
    # plt.xticks(rotation=90)
    # yticks = [i for i in range(0, max(labels.count['agreement_number']) + 1, 1)]
    # plt.yticks(yticks)
    plt.show()
    return labels


def calculate_not_relevant(df):
    df = df[df['label'] == 'not relevant']
    # check the distribution by the query column
    not_relevant_df = df['query number'].value_counts()
    not_relevant_df = not_relevant_df.to_frame()
    not_relevant_df = not_relevant_df.reset_index()
    not_relevant_df = not_relevant_df.rename(columns={'index': 'query number', 'query number': 'count'})
    not_relevant_df['percentage'] = not_relevant_df['count'] / not_relevant_df['count'].sum()
    not_relevant_df['percentage'] = not_relevant_df['percentage'].apply(lambda x: "{:.2%}".format(x))
    not_relevant_df = not_relevant_df.sort_values(by=['count'], ascending=False)
    not_relevant_df = not_relevant_df.reset_index(drop=True)

    old_df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Data',
                     'spike_results.csv'))
    # check how many of the rows of not_relevant_df are in the old_df
    mask_old = df[df['sentence_id'].isin(old_df['sentence_id'])]

    # add to df the column of origin_sentence from the old_df, where the sentence_id is the same
    df.drop_duplicates(subset=['sentence_id'], keep='first', inplace=True)
    old_df.drop_duplicates(subset=['sentence_id'], keep='first', inplace=True)
    org_df = df.merge(old_df[['sentence_id', 'origin_sentence']], on='sentence_id')

    new_df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'Data',
                     'spike_results_v2.csv'))
    mask_new = org_df[org_df['origin_sentence'].isin(new_df['origin_sentence'])]

    return df


def get_stats_on_len_of_contexts_gray(df):
    plt.rcParams['font.size'] = 14
    df['start_ind'] = df['start_ind'].astype('int')
    df['end_ind'] = df['end_ind'].astype('int')
    df['len_of_context'] = df['end_ind'] - df['start_ind'] - 1
    threshold0 = 10
    threshold1 = 15
    threshold2 = 20
    df['grouped_value'] = df['len_of_context'].apply \
        (lambda x: ('>{}'.format(threshold0) if (threshold1 > x >= threshold0) else
                    ('>{}'.format(threshold1) if threshold2 > x >= threshold1 else
                     ('>{}'.format(threshold2) if x >= threshold2 else
                      x))))

    # len_of_context = df['grouped_value'].value_counts()
    # plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(5.9, 5.9))
    # df.groupby('grouped_value').count()['len_of_context'].plot(kind='bar', width=0.5, color='blue')
    data = df.groupby('grouped_value').count()['len_of_context']

    encoded_x = data.index.astype('category').codes
    bars = ax.bar(encoded_x, data.values, color='lightgray')
    plt.setp(ax.spines.values(), color='lightgray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    ax.set_xticks(encoded_x)
    ax.set_xticklabels(data.index)
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

    # for bar in bars:
    #     bar.set_edgecolor("black")
    #     bar.set_linewidth(2)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')

    plt.xlabel('Number of Sentences Revealed\n(Excluding Selected Sentence)', fontsize=11)
    plt.ylabel('Number of Labeled Samples', fontsize=11)
    # rorate the x labels
    plt.xticks(rotation=0, fontsize=10, color='gray')
    plt.yticks(fontsize=12, color='gray')
    # plt.title("Amount of Context Needed for\n Accurate Sentence Classification", fontsize=12)
    # plt.grid(visible=True, axis='y', linestyle='--', linewidth=0.5)
    # plt.grid(visible=True, axis='x', linestyle='--', linewidth=0.5)
    # very high resolution
    # tight_layout()
    plt.tight_layout()
    # plt.savefig(os.path.join(DATA_STATS_FOLDER, 'len_of_context.png'), dpi=1200)
    plt.savefig('len_of_context_gray.png', dpi=2400)
    # plt.savefig('len_of_context.pdf')
    # plt.savefig('len_of_context.svg', format='svg', dpi=1200)
    plt.show()


def get_stats_on_len_of_contexts(df):
    plt.rcParams['font.size'] = 14
    df['start_ind'] = df['start_ind'].astype('int')
    df['end_ind'] = df['end_ind'].astype('int')
    df['len_of_context'] = df['end_ind'] - df['start_ind'] - 1
    threshold0 = 10
    threshold1 = 15
    threshold2 = 20
    df['grouped_value'] = df['len_of_context'].apply \
        (lambda x: ('>{}'.format(threshold0) if (threshold1 > x >= threshold0) else
                    ('>{}'.format(threshold1) if threshold2 > x >= threshold1 else
                     ('>{}'.format(threshold2) if x >= threshold2 else
                      x))))

    # len_of_context = df['grouped_value'].value_counts()
    # plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(5.9, 5.9))
    # df.groupby('grouped_value').count()['len_of_context'].plot(kind='bar', width=0.5, color='blue')
    data = df.groupby('grouped_value').count()['len_of_context']

    encoded_x = data.index.astype('category').codes
    bars = ax.bar(encoded_x, data.values, color='orange')

    ax.set_xticks(encoded_x)
    ax.set_xticklabels(data.index)

    for bar in bars:
        bar.set_edgecolor("black")
        bar.set_linewidth(2)

    plt.xlabel('Number of Sentences Revealed\n(Excluding Selected Sentence)', fontsize=11)
    plt.ylabel('Number of Labeled Samples', fontsize=11)
    # rorate the x labels
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=12)
    # plt.title("Amount of Context Needed for\n Accurate Sentence Classification", fontsize=12)
    # plt.grid(visible=True, axis='y', linestyle='--', linewidth=0.5)
    # plt.grid(visible=True, axis='x', linestyle='--', linewidth=0.5)
    # very high resolution
    # tight_layout()
    plt.tight_layout()
    # plt.savefig(os.path.join(DATA_STATS_FOLDER, 'len_of_context.png'), dpi=1200)
    plt.savefig('len_of_context.png', dpi=1200)
    # plt.savefig('len_of_context.pdf')
    # plt.savefig('len_of_context.svg', format='svg', dpi=1200)
    plt.show()




def plot_label_distribution_train(df):
    df = df[['sentence_id', 'label']]
    # drop label starting with 'O'
    df = df[df['label'].str.startswith('"אמינה אבל"') == False]
    map_labels_file = os.path.join(DATA_FOLDER, 'Old_Categories.csv')
    map_labels = pd.read_csv(map_labels_file)
    map_labels = map_labels.set_index('category')
    # add not relevant category
    df['parent_category_label'] = df['label'].apply(lambda x: map_labels.loc[x]['parent_category'])
    df['parent_category_label'] = df['parent_category_label'].apply(lambda x: x[::-1] if not x.startswith('not') else x)
    df.drop('label', axis=1, inplace=True)
    count = df.groupby('parent_category_label').count()

    # df['label'] = df['label'].apply(lambda x: x[::-1] if not x.startswith('not') else x)
    # count = df.groupby('label').count()

    count.rename(columns={'sentence_id': 'count'}, inplace=True)
    plot_label_distribution(count, num_sentences=len(df))
    return count


def get_label_distribution_test(df, score=0.):
    df = df[df['score_0'] > score]
    df = df[['sentence_id', 'label_0']]
    df.rename(columns={'label_0': 'label'}, inplace=True)

    map_labels_file = os.path.join(DATA_FOLDER, 'Old_Categories.csv')
    map_labels = pd.read_csv(map_labels_file)
    map_labels = map_labels.set_index('category')
    # add not relevant category
    df['parent_category_label'] = df['label'].apply(lambda x: map_labels.loc[x]['parent_category'])
    df['parent_category_label'] = df['parent_category_label'].apply(lambda x: x[::-1] if not x.startswith('not') else x)
    df.drop('label', axis=1, inplace=True)
    count = df.groupby('parent_category_label').count()

    # df['label'] = df['label'].apply(lambda x: x[::-1] if not x.startswith('not') else x)
    # count = df.groupby('label').count()

    # rename the column
    count.rename(columns={'sentence_id': 'count'}, inplace=True)
    return count



def plot_label_distribution(count, num_sentences, file_name='train_label_distribution.png'):
    # take the 200 first rows
    # df = df.head(200)
    # sort the labels by the number of sentences
    # sort pd.series by
    count = count.sort_values(by='count', ascending=False)
    labels = count.index
    count = count['count']
    # round the values to 1 decimal
    count = count.apply(lambda x: round(x, 1))

    # plot the number of sentences for each label, with y label as the number of sentences, and x label as the label.
    # use malplotlib to plot the graph and not pandas
    plt.figure(figsize=(12, 8))
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.6)  # <-- Change the 0.02 to work for your plot.
    ax = fig.add_subplot(111)
    ax.bar(labels, count)
    ax.set_ylabel('Average Number of sentences')
    ax.set_xlabel('Label')
    ax.set_title(f'Train Dataset \n Number of sentences for each label (total number of sentences: {num_sentences})')
    plt.xticks(rotation=90)
    # yticks = [i for i in range(0, max(count) + 1, 1)]
    # plt.yticks(yticks)
    # add the number of sentences for each label
    for i, v in enumerate(count):
        # add the percentage of the total number of sentences
        text = f'{v} ({round(v / num_sentences * 100, 2)}%)'
        ax.text(i - 0.25, v + 0.1, str(text), color='blue', fontweight='bold')

    plt.savefig(file_name, dpi=300)
    # plt.show()


def check_dist_on_files(df):
    df = df[df['label'] != 'not relevant']
    map_labels_file = os.path.join(DATA_FOLDER, 'Old_Categories.csv')
    map_labels = pd.read_csv(map_labels_file)
    map_labels = map_labels.set_index('category')
    # add not relevant category
    df['parent_category_label'] = df['label'].apply(lambda x: map_labels.loc[x]['parent_category'])
    df['parent_category_label2'] = df['label2'].apply(
        lambda x: map_labels.loc[x]['parent_category'] if not pd.isna(x) else x)

    # groupby title (file name) and check if there is a file with more than one sentence,
    # and if so, check if the labels are the same
    df2 = df[['title', 'sentence_id', 'label', 'label2', 'parent_category_label', 'parent_category_label2']]
    # gropu by title,and agg all the labels to a list
    df2 = df2.groupby('title').agg({'sentence_id': lambda x: list(x), 'label': lambda x: list(x),
                                    'label2': lambda x: list(x), 'parent_category_label': lambda x: list(x),
                                    'parent_category_label2': lambda x: list(x)})
    # take only the files with more than one sentence
    df2 = df2[df2['sentence_id'].apply(lambda x: len(x) > 1)]
    df3 = df2[df2['sentence_id'].apply(lambda x: 3 > len(x) > 1)]
    # count the row that that have the same labels in the list
    df3['same_labels'] = df3.apply(lambda x: len(set(x['label'])) == 1, axis=1)
    df3['same_parent_category'] = df3.apply(lambda x: len(set(x['parent_category_label'])) == 1, axis=1)
    # get the percentage of files with the same labels
    print(f'Percentage of files with the same labels: {df3["same_labels"].sum() / len(df3)}')
    print(f'Percentage of files with the same parent category: {df3["same_parent_category"].sum() / len(df3)}')


def correct_not_relevant(df, shared_df):
    df = df[~df['sentence_id'].isin(shared_df['sentence_id'])]

    df_withput_not_relevant = df[df['label'] != 'not relevant']
    not_relevant = len(df[df['label'] == 'not relevant'])
    df_withput_not_relevant[df_withput_not_relevant['query number'].isnull()].sum()
    correct_not_relevant = len(df_withput_not_relevant[df_withput_not_relevant['query number'].isnull()])

    print(f'Number of not relevant sentences: {not_relevant}')
    print(f'Number of not relevant sentences that correct: {correct_not_relevant}')


def train_main():
    df = read_annotated_data()
    shared_df = get_shared_sentences()

    df = df[df['status'] == 'annotated']

    correct_not_relevant(df, shared_df)

    # drop sentences that are in shared_df
    # drop not relevant sentences
    df = df[df['label'] != 'not relevant']
    df.drop_duplicates(subset=['sentence_id'], keep='first', inplace=True)
    check_dist_on_files(df)
    df = df[~df['sentence_id'].isin(shared_df['sentence_id'])]

    count = plot_label_distribution_train(df)

    get_stats_on_len_of_contexts_gray(df)
    return count


def test_main():
    ckpt = 120
    counts = []
    lens_of_df = []
    for i in range(3):
        result_name = f'predictions_classify_datasets5_seed{i}_checkpoint-{ckpt}'
        path = r'C:\Users\t-eliyahabba\OneDrive-Microsoft\Documents\classify_models2\results\classify_datasets5_seed{}'.format(
            i)
        df = pd.read_csv(os.path.join(path, f'{result_name}.csv'))
        count = get_label_distribution_test(df, score=0.5)
        counts.append(count)
        lens_of_df.append(sum(count['count']))
    # calculate the average of the counts for each label
    counts = pd.concat(counts, axis=1)
    # calculate the average but save the name of the column
    counts = counts.mean(axis=1).to_frame()
    counts.rename(columns={0: 'count'}, inplace=True)
    plot_label_distribution(counts, int(sum(lens_of_df) / len(lens_of_df)),
                                 file_name='high_level_distribution_test_score_0.5.png')

    return counts



def cal_kl_divergence(p, q):
    # Calculate the KL divergence
    from scipy.stats import entropy
    train_data_category = p
    test_data_category = q
    kl_divergence = entropy(train_data_category, test_data_category)
    print(f'KL divergence: {kl_divergence}')


if __name__ == '__main__':
    df_train = train_main()
    calculate_not_relevant(df_train)
    # df_test = test_main()
    # cal_kl_divergence(df_train['count'], df_test['count'])
