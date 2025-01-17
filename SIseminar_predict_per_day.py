import matplotlib.pyplot as plt
import datetime as dt
from glob import glob
from tqdm import tqdm
import json
import pandas as pd
import re
import csv
import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor

### dfに追加するlistを返す
def get_train_list(project, PR_list, madelist, finlist, current_date, the_next_date, remaining_release_days):
    metrics_list = []
    objective_list = []
    PR_num = len(PR_list)
    review_lines = 0
    for PR_json in PR_list:
        try:
            name = str(PR_json['_number'])
            message_list = PR_json['messages']
            revs = commit_number(message_list, current_date)
            with open(f'/Users/mizuki-u/current/dataset/{project}/revision/hash/{name}_rev{revs}.json') as f:
                com_msg_load = json.load(f)
            with open(f'/Users/mizuki-u/current/dataset/{project}/revision/diff_lines/{name}_rev{revs}.json') as f:
                rev_size_load = json.load(f)
            #### 抽出再開
            accept_rate, finlist = calc_accept_rate(PR_json, finlist)
            PRs, madelist = PR_count(PR_json, madelist)
            additions = additions_num(rev_size_load)
            deletions = deletions_num(rev_size_load)
            has_test_code = Has_Test_Code(rev_size_load)
            is_bug = Is_Bug(com_msg_load)
            is_SAR = Is_SA_Refactoring(com_msg_load)
            metrics_list.append([accept_rate, PRs, additions, deletions, has_test_code, is_bug, is_SAR, remaining_release_days, PR_num])
            TF_review = first_review_in_span(message_list, current_date, the_next_date) - 1
            objective_list.append(TF_review)
            if TF_review == 1:
                review_lines += additions + deletions
        except FileNotFoundError:
            pass
    for PR_metrics in metrics_list:
        PR_metrics.append(review_lines)
    return metrics_list, objective_list

def get_PR_list(create_sort_list, project, current_date, the_next_date):
    Not_review_PR_list = []
    made_PRs_list = [['null', 0]]
    fin_PRs_list = [['null', 0, 0]]
    delta_half_year = dt.timedelta(days=182)
    except_data_day = current_date - delta_half_year
    for number in create_sort_list:
        with open(f'/Users/mizuki-u/current/dataset/{project}/list_1/{number}.json') as f:
            json_load = json.load(f)
        open_time = pd.to_datetime(json_load['created']).normalize()
        message_list = json_load['messages']
        close_time = merged_or_abandoned_date(message_list)
        if(open_time < except_data_day):
            continue
        if(open_time < current_date):
            ### ここからmade_PR
            try:
                owner = json_load['owner']['name']
            except KeyError:
                owner = 'null'
            flag1 = 0
            for owner_PRs in made_PRs_list:
                if owner_PRs[0] == owner:
                    owner_PRs[1] += 1
                    flag1 += 1
                    break
            if flag1 == 0:
                made_PRs_list.append([owner, 1])
            ### ここからfin_PR
            if close_time < current_date:
                flag3 = 0
                status = json_load['status']
                for owner_PRs in fin_PRs_list:
                    if owner_PRs[0] == owner:
                        if status == 'MERGED':
                            owner_PRs[1] += 1
                        else:
                            owner_PRs[2] += 1
                        flag3 += 1
                        break
                if flag3 == 0:
                    if status == 'MERGED':
                        fin_PRs_list.append([owner, 1, 0])
                    else:
                        fin_PRs_list.append([owner, 0, 1])
        if((current_date <= close_time) and (open_time < the_next_date)):
            review_judge = first_review_in_span(message_list, current_date, the_next_date)
            if review_judge != 0:
                Not_review_PR_list.append(json_load)
        if(the_next_date < open_time):
            break
    return Not_review_PR_list, made_PRs_list, fin_PRs_list

def create_sort(filepath_list):
    create_sort_list = []
    for filepath in filepath_list:
        with open(filepath) as f:
            json_load = json.load(f)
        create_sort_list.append(json_load['_number'])
    return sorted(create_sort_list)

def merged_or_abandoned_date(message_list):
    flag = False
    for msg in message_list:
        if re.search('Change has been successfully merged', msg['message'], re.IGNORECASE):
            return pd.to_datetime(msg['date']).normalize()
        elif re.match('Abandoned', msg['message'], re.IGNORECASE):
            flag = True
            ret_msg = pd.to_datetime(msg['date']).normalize()
    if flag == True:
        return ret_msg
    else:
        return pd.to_datetime('2025-01-01 00:00:00.000000000').normalize()

#-------
### PR作成者の過去に作成したPRのマージ率
def calc_accept_rate(json_load, finlist):
    try:
        owner = json_load['owner']['name']
    except KeyError:
        owner = 'null'
    status = json_load['status']
    per_merge = 0
    for owner_list in finlist:
        if(owner == owner_list[0]):
            per_merge = (owner_list[1] / (owner_list[1] + owner_list[2])) * 100
            if(status == 'MERGED'):
                owner_list[1] += 1
            else:
                owner_list[2] += 1
            break
    return per_merge, finlist

### 追加行数
def additions_num(rev_load):
    count = 0
    for file in rev_load:
        additions = rev_load[file].get('lines_inserted', 0)
        count += additions    
    return count

### 削除行数
def deletions_num(rev_load):
    count = 0
    for file in rev_load:
        deletions = rev_load[file].get('lines_deleted', 0)
        count += deletions    
    return count

#-------
### コミット数（リビジョン数）
def commit_number(message_list, current_date):
    revision = 1
    for msg in message_list:
        date = pd.to_datetime(msg['date']).normalize()
        if(date <= current_date):
            revision = msg['_revision_number']
        else:
            break
    return revision

### ファイル数（プログラム数）
def file_number(rev_load):
    count = 0
    for _ in rev_load:
        count += 1
    return count

### レビューコメント数
def review_comment_count(message_list, current_date):
    count = 0
    for msg in message_list:
        date = pd.to_datetime(msg['date']).normalize()
        if(date < current_date):
            if any(['Looks good to me, approved' in msg['message'], 'Looks good to me, but someone else must approve' in msg['message'],
                    'I would prefer this is not submitted as is' in msg['message'], 'This shall not be submitted' in msg['message'],
                    'Code-Review+2' in msg['message'], 'Code-Review+1' in msg['message'], 'Code-Review-1' in msg['message'],
                    'Code-Review-2' in msg['message']]):
                count += 1
            elif '-Code-Review' in msg['message']:
                count -= 1
        else:
            break
    return count

#-------
### PR作成者の過去に作成したPR数
def PR_count(json_load, madelist):
    try:
        owner = json_load['owner']['name']
    except KeyError:
        owner = 'null'
    PRs = 0
    for simplelist in madelist:
        if(owner == simplelist[0]):
            PRs = simplelist[1]
            simplelist[1] += 1
            break
    return PRs, madelist
            
### テストコードがファイルにあるか
def Has_Test_Code(rev_load):
    for file in rev_load:
        if re.search(r'test|spec', file, re.IGNORECASE):
            return 1
    return 0

### bugかどうか
def Is_Bug(msg_load):
    confidence_level = 0
    title_description = msg_load['message']
    modified_text = re.sub(r'Change-Id.*?\n', '', title_description)
    pattern = re.compile(r'[^a-zA-Z0-9]')
    clean_text = pattern.sub('', modified_text)
    regex_patterns = [
        r"bug[# \t]*[0-9]+",
        r"pr[# \t]*[0-9]+",
        r"show\_bug\.cgi\?id=[0-9]+",
        r"\[[0-9]+\]"
    ]
    for pattern in regex_patterns:
        if re.search(pattern, clean_text, re.IGNORECASE):
            confidence_level += 1
            if re.fullmatch(pattern, clean_text, re.IGNORECASE):
                confidence_level += 1
                return confidence_level
            break
    if any([re.search(r'fix(e[ds])?|bugs?|defects?|patch', clean_text, re.IGNORECASE),
            re.fullmatch(r"[0-9]+", clean_text, re.IGNORECASE)]):
        confidence_level += 1
    return confidence_level

### sar（self-affirmedリファクタリング）かどうか
def Is_SA_Refactoring(msg_load):
    title_description = msg_load['subject']
    modified_text = re.sub(r'Change-Id.*?\n', '', title_description)
    clean_text = modified_text.replace('\n', ' ')
    patterns = [r'refactor*', r'mov*', r'split*', r'introduc*', r'decompos*',
                r'reorganiz*', r'extract*', r'merg*', r'renam*', r'chang*', r'restructur*',
                r'reformat*', r'extend*', r'remov*', r'replac*', r'rewrit*', r'simplif*',
                r'creat*', r'improv*', r'add*', r'modif*', r'enhanc*', r'rework*',
                r'inlin*', r'redesign*', r'reduc*', r'encapsulat*',
                r'cleanup', r'removed poor coding practice', r'improve naming consistency',
                r'removing unused classes', r'pull some code up', r'use better name',
                r'replace it with', r'make maintenance easier', r'code cleanup',
                r'minor Simplification', r'reorganize project structures',
                r'code maintenance for refactoring', r'remove redundant code',
                r'moved and gave clearer names to', r'refactor bad designed code',
                r'getting code out of', r'deleting a lot of old stuff', r'code revision',
                r'fix technical debt', r'fix quality issue', r'antipattern bad for performances',
                r'major/minor structural changes', r'clean up unnecessary code',
                r'code reformatting & reordering', r'nicer code / formatted / structure',
                r'simplify code redundancies', r'added more checks for quality factors',
                r'naming improvements', r'renamed for consistency', r'refactoring towards nicer name analysis',
                r'change design', r'modularize the code', r'code cosmetics', r'moved more code out of',
                r'remove dependency', r'enhanced code beauty', r'simplify internal design',
                r'change package structure', r'use a safer method', r'code improvements', r'minor enhancement',
                r'get rid of unused code', r'fixing naming convention', r'fix module structure', r'code optimization',
                r'fix a design flaw', r'nonfunctional code cleanup', r'improve code quality', r'fix code smell',
                r'use less code', r'avoid future confusion', r'more easily extended', r'polishing code',
                r'move unused file away', r'many cosmetic changes', r'inlined unnecessary classes',
                r'code cleansing', r'fix quality flaws', r'simplify the code']
    for i, pattern in enumerate(patterns):
        if re.search(pattern, clean_text, re.IGNORECASE):
            return 1
    return 0


def review_in_span(message_list, current_date, the_next_date):
    for msg in message_list:
        date = pd.to_datetime(msg['date']).normalize()
        if current_date <= date < the_next_date:
            if any(['Looks good to me, approved' in msg['message'], 'Looks good to me, but someone else must approve' in msg['message'],
                    'I would prefer this is not submitted as is' in msg['message'], 'This shall not be submitted' in msg['message'],
                    'Code-Review+2' in msg['message'], 'Code-Review+1' in msg['message'], 'Code-Review-1' in msg['message'],
                    'Code-Review-2' in msg['message']]):
                return 1
        if the_next_date <= date:
            return 0
    return 0


def first_review_in_span(message_list, current_date, the_next_date):
    for msg in message_list:
        date = pd.to_datetime(msg['date']).normalize()
        if any(['Looks good to me, approved' in msg['message'], 'Looks good to me, but someone else must approve' in msg['message'],
                    'I would prefer this is not submitted as is' in msg['message'], 'This shall not be submitted' in msg['message'],
                    'Code-Review+2' in msg['message'], 'Code-Review+1' in msg['message'], 'Code-Review-1' in msg['message'],
                    'Code-Review-2' in msg['message']]):
            if date < current_date:
                return 0
            elif current_date <= date < the_next_date:
                return 2
        if the_next_date <= date:
            return 1
    return 1

    
def predict_release(m_train, m_test, o_train, o_test, project):    
    propose_model = RandomForestRegressor(random_state=0)
    propose_model.fit(m_train, o_train)
    propose_F1_list = []

    for metrics, objective in zip(m_test, o_test):
        propose_test_pred = propose_model.predict(metrics)
        propose_F1_list.append(f1_score(objective, propose_test_pred))


def window_slide(project, create_sort_list, ver_num, end_date, current_date, the_next_date):
    review_all_metrics_list = []
    review_all_objective_list = []

    remaining_release_days = 84
    while the_next_date <= end_date:
        Not_review_PR_list, made_PR_list, fin_PR_list = get_PR_list(create_sort_list, project, current_date, the_next_date)
        review_metrics_list, review_TF_list = get_train_list(project, Not_review_PR_list, made_PR_list, fin_PR_list, 
                                                            current_date, the_next_date, remaining_release_days)
        if ver_num != 4:
            review_all_metrics_list.extend(review_metrics_list)
            review_all_objective_list.extend(review_TF_list)
        else:
            review_all_metrics_list.append(review_metrics_list)
            review_all_objective_list.append(review_TF_list)

        current_date += pd.Timedelta(days=1)
        the_next_date += pd.Timedelta(days=1)
        remaining_release_days -= 1
    return review_all_metrics_list, review_all_objective_list


## 成熟後かつ3ヶ月の期間が被っていないバージョン
# project_list = ['Glance', 'Swift', 'Keystone', 'Cinder', 'Neutron', 'Nova']
# dates_list = [[dt.datetime(2015,6,18), dt.datetime(2015,12,2), dt.datetime(2016,3,3), dt.datetime(2016,6,1), dt.datetime(2018,4,19)],
#               [dt.datetime(2013,8,20), dt.datetime(2015,9,1), dt.datetime(2016,3,25), dt.datetime(2018,2,5), dt.datetime(2021,3,18)],
#               [dt.datetime(2015,6,17), dt.datetime(2016,3,3), dt.datetime(2016,7,14), dt.datetime(2016,11,17), dt.datetime(2019,3,21)],
#               [dt.datetime(2015,12,3), dt.datetime(2016,3,17), dt.datetime(2016,7,14), dt.datetime(2017,6,8), dt.datetime(2017,12,6)],
#               [dt.datetime(2015,6,24), dt.datetime(2015,12,3), dt.datetime(2016,6,3), dt.datetime(2017,4,14), dt.datetime(2020,2,22)],
#               [dt.datetime(2016,3,3), dt.datetime(2016,6,2), dt.datetime(2017,6,8), dt.datetime(2017,10,24), dt.datetime(2019,9,27)]]
# release_vers_list = [['11.0.0a0', '12.0.0.0b1', '12.0.0.0b3', '13.0.0.0b1', '17.0.0.0b1'], ['1.9.2', '2.4.0', '2.7.0', '2.17.0', '2.27.0'],
#                      ['8.0.0a0', '9.0.0.0b3', '10.0.0.0b2', '11.0.0.0b1', '15.0.0.0rc1'], ['8.0.0.0b1', '8.0.0.0rc1', '9.0.0.0b2', '11.0.0.0b2', '12.0.0.0b2'],
#                      ['7.0.0.0b1', '8.0.0.0b1', '9.0.0.0b1', '11.0.0.0b1', '16.0.0.0b1'], ['13.0.0.0b3', '14.0.0.0b1', '16.0.0.0b2', '17.0.0.0b1', '20.0.0.0rc1']]

## 一番サイズの小さいプロジェクトで動作確認
project_list = ['Glance']
dates_list = [[dt.datetime(2015,6,18), dt.datetime(2015,12,2), dt.datetime(2016,3,3), dt.datetime(2016,6,1), dt.datetime(2018,4,19)]]
release_vers_list = [['11.0.0a0', '12.0.0.0b1', '12.0.0.0b3', '13.0.0.0b1', '17.0.0.0b1']]

time_delta = dt.timedelta(weeks=12)

for project, dates, release_vers in tqdm(zip(project_list, dates_list, release_vers_list)):
    filepath_list = glob(f'/Users/mizuki-u/current/dataset/{project}/list_1/*.json')
    create_sort_list = create_sort(filepath_list)
    review_two_weeks_train_metrics_list = []
    review_two_weeks_train_objective_list = []

    for ver_num, (date, release_ver) in enumerate(zip(dates, release_vers)):
        start_date = (date - time_delta).date()
        end_date = date.date()
        current_date = start_date
        
        the_next_date = current_date + pd.Timedelta(days=14)
        if ver_num != 4:
            review_metrics_list, review_objective_list = window_slide(project, create_sort_list, ver_num, end_date, current_date, the_next_date)
            review_two_weeks_train_metrics_list.extend(review_metrics_list)
            review_two_weeks_train_objective_list.extend(review_objective_list)
        else:
            review_two_weeks_test_metrics_list, review_two_weeks_test_objective_list = window_slide(project, create_sort_list, ver_num, end_date, current_date, the_next_date)

    predict_release(review_two_weeks_train_metrics_list, review_two_weeks_test_metrics_list,
                    review_two_weeks_train_objective_list, review_two_weeks_test_objective_list, project)