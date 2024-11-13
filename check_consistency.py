from sklearn.metrics import normalized_mutual_info_score

def calculate_nmi(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)