def extract_features(pr_data, start_date, end_date):
    """PRデータから特徴量を抽出"""
    metrics_list = []
    objective_list = []

    for pr in pr_data:
        # 特徴量の作成
        metrics_list.append([
            len(pr.get('messages', [])),  # メッセージ数
            pr.get('lines_inserted', 0), # 追加行数
            pr.get('lines_deleted', 0)   # 削除行数
        ])
        
        # PR内のメッセージを評価してラベルを決定
        is_positive = any(
            'Looks good to me, approved' in msg.get('message', '') or
            'Looks good to me, but someone else must approve' in msg.get('message', '') or
            'I would prefer this is not submitted as is' in msg.get('message', '') or
            'This shall not be submitted' in msg.get('message', '') or
            'Code-Review+2' in msg.get('message', '') or
            'Code-Review+1' in msg.get('message', '') or
            'Code-Review-1' in msg.get('message', '') or
            'Code-Review-2' in msg.get('message', '')
            for msg in pr.get('messages', [])
        )
        objective_list.append(1 if is_positive else 0)

    return metrics_list, objective_list
