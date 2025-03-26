from collections import defaultdict
import pandas as pd
import math
import sys
import joblib
import os

attributes = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]


def preprocess(record):
    transaction = set()
    class_name = None  
    
    # Create ranges for numerical features
    for attr, val in zip(attributes, record):
        if attr == "age":
            val = int(val)
            if 0 < val <= 20:
                transaction.add(f"{attr}=<=20")
            elif 20 < val <= 30:
                transaction.add(f"{attr}=[21,30]")
            elif 30 < val <= 40:
                transaction.add(f"{attr}=[31,40]")
            elif 40 < val <= 50:
                transaction.add(f"{attr}=[41,50]")
            elif 50 < val <= 60:
                transaction.add(f"{attr}=[51,60]")
            else:
                transaction.add(f"{attr}=>61")
        
        elif attr == "fnlwgt":
            val = int(val)
            if 0 < val <= 50000:
                transaction.add(f"{attr}=<=50000")
            elif 50001 <= val <= 150000:
                transaction.add(f"{attr}=[50001,150000]")
            elif 150001 <= val <= 250000:
                transaction.add(f"{attr}=[150001,250000]")
            elif 250001 <= val <= 350000:
                transaction.add(f"{attr}=[250001,350000]")
            else:
                transaction.add(f"{attr}=>350000")

        elif attr == "education-num":
            val = int(val)
            if 0 <= val <= 4:
                transaction.add(f"{attr}=<=4")
            elif 5 <= val <= 8:
                transaction.add(f"{attr}=[5,8]")
            elif 9 <= val <= 12:
                transaction.add(f"{attr}=[9,12]")
            else:
                transaction.add(f"{attr}=>12")
            
        elif attr == "capital-gain":
            val = int(val)
            if val == 0:
                transaction.add(f"{attr}=0")
            elif 1 <= val <= 1000:
                transaction.add(f"{attr}=[1,1000]")
            else:
                transaction.add(f"{attr}=>1000")
        
        elif attr == "capital-loss":
            val = int(val)
            if val == 0:
                transaction.add(f"{attr}=0")
            elif 1 <= val <= 1000:
                transaction.add(f"{attr}=[1,1000]")
            else:
                transaction.add(f"{attr}=>1000")
        
        elif attr == "hours-per-week":
            val = int(val)
            if 0 <= val <= 15:
                transaction.add(f"{attr}=<=15")
            elif 16 <= val <= 30:
                transaction.add(f"{attr}=[16,30]")
            elif 31 <= val <= 45:
                transaction.add(f"{attr}=[31,45]")
            elif 46 <= val <= 60:
                transaction.add(f"{attr}=[46,60]")
            elif 61 <= val <= 75:
                transaction.add(f"{attr}=[61,75]")
            else:
                transaction.add(f"{attr}=>75")
        
        elif attr == "income": 
            class_name = str(val).strip()

        else: # Other features
            val = str(val).strip()
            transaction.add(f"{attr}={val}")
    return transaction, class_name

def read_transactions(data):
    class_transactions = defaultdict(list)
    for record in data:
        transaction, class_label = preprocess(record)
        if class_label:
            class_transactions[class_label].append(transaction)
    return class_transactions

def get_min_support(total_transactions, percentage):
    return math.ceil(total_transactions * percentage)

def build_vertical_format(transactions):
    vertical = defaultdict(set)
    for tid, transaction in enumerate(transactions, start=1):
        for item in transaction:
            vertical[item].add(tid)
    return vertical

def is_closed(itemset, tidset, closed_itemsets):
    for item in closed_itemsets:
        if itemset < item and tidset == closed_itemsets[item]['tidset']: # If itemset is covered by another item in closed_itemsets
            return False
    return True

def subsumption_check(C, closure, closure_tidset):
    if is_closed(closure, closure_tidset, C):
        C[frozenset(closure)] = {
            'tidset': closure_tidset,
            'support': len(closure_tidset)
        }

def charm_property(X, Xi, Xj, Pi, vertical, min_support, P):
    if len(X) >= min_support: 
        if vertical[Xj] == X:  
            Pi.add(Xj)
            P.remove(Xj)
            Xi.add(Xj)
        elif vertical[Xj] > X:  
            Pi.add(Xj)
            Xi.add(Xj)

def charm_extend(prefix, P, C, vertical, min_support):
    while P:
        li = P.pop(0)
        li_tidset = vertical[li]
        Pi = prefix.union({li})
        
        closure_tidset = li_tidset.copy()
        for item in Pi:
            closure_tidset &= vertical[item]

        closure = set()
        closure = Pi.union(closure)
        new_P = []

        for lj in P:
            intersection = closure_tidset & vertical[lj]
            if len(intersection) >= min_support:
                new_P.append(lj)
                charm_property(closure_tidset, closure, lj, Pi, vertical, min_support, new_P)
        
        if closure:
            closure_tidset = set.intersection(*(vertical[item] for item in closure))
        else:
            closure_tidset = set()
        
        if len(closure_tidset) >= min_support:
            subsumption_check(C, closure, closure_tidset)

        charm_extend(Pi, new_P, C, vertical, min_support)
       

def charm(transactions, min_support):
    vertical = build_vertical_format(transactions)
    
    vertical = {item: tids for item, tids in vertical.items() if len(tids) >= min_support}

    sorted_items = sorted(vertical.keys(), key=lambda item: len(vertical[item]), reverse=True)
    
    closed_itemsets = dict()
    
    charm_extend(set(), sorted_items, closed_itemsets, vertical, min_support)
    
    return closed_itemsets

def calculate_support_score(closed_itemsets, class_transactions):
    # Build vertical format for each class
    class_vertical = {label: build_vertical_format(transactions) for label, transactions in class_transactions.items()}
    
    frequency = defaultdict(int)  
    support_scores = {}

    for itemset, data in closed_itemsets.items():
        class_supports = {}
        
        for class_label, vertical in class_vertical.items():

            class_support = len(set.intersection(*(vertical[item] for item in itemset if item in vertical)))
            class_supports[class_label] = class_support 
            frequency[itemset] += class_support  
        
        support_scores[itemset] = class_supports

    for itemset in support_scores:
        for class_label in support_scores[itemset]:
            if frequency[itemset] != 0:
                support_scores[itemset][class_label] /= frequency[itemset]
            else:
                support_scores[itemset][class_label] = 0
    
    return support_scores

def find_class_explainable_itemset(class_itemsets, support_scores):
    explainable_itemset = {}
    max_class_score = {}
    
    for class_label, itemsets in class_itemsets.items():
        max_score = 0
        for itemset in itemsets:
            score = support_scores.get(frozenset(itemset), {}).get(class_label, 0)
            if (score > max_score):
                max_score = score
                explainable_itemset[class_label] = itemset
                max_class_score[class_label] = max_score       

    return explainable_itemset, max_class_score

def classify(transaction, class_itemsets, support_scores):
    class_matched_itemsets = {}  
    class_matched_counts = {} 
    class_scores = {} 
    explainable_itemset = {}

    predicted_class = None
    max_total_score = -1

    for class_label, itemsets in class_itemsets.items():
        matched_itemsets = []  
        total_score = 0
        max_score = 0
        for itemset in itemsets:
            if itemset.issubset(transaction):
                matched_itemsets.append(itemset) 
                score = support_scores.get(frozenset(itemset), {}).get(class_label, 0)
                total_score += score
                if (score > max_score):
                    max_score = score
                    explainable_itemset[class_label] = itemset        
        
        class_matched_itemsets[class_label] = matched_itemsets
        class_scores[class_label] = total_score
        
        if (total_score > max_total_score):
            max_total_score = total_score
            predicted_class = class_label
    
    # Uncomment to print instance-wise explanations
    # print("Explanation: ", explainable_itemset[predicted_class])

    return predicted_class, class_matched_itemsets, class_matched_counts

def prepare_data_for_mlp(encoder, df):
    feature_columns = attributes[:-1]
    target_column = 'income'
    
    X = df[feature_columns]
    y = df[target_column]
    
    X_encoded = encoder.transform(X)    
    return X_encoded, y

def compare_mlp_charm(test_df, mlp, encoder, class_itemset, support_scores):
    X_test_encoded, y_test = prepare_data_for_mlp(encoder, test_df)
    
    mlp_predictions = mlp.predict(X_test_encoded)
    
    charm_predictions = []
    for record in test_df.values:
        transaction, _ = preprocess(record)
        predicted_class, _, _ = classify(transaction, class_itemset, support_scores)
        charm_predictions.append(predicted_class)

    count = 0
    count2 = 0
    count3 = 0
   
    for i, (mlp_pred, charm_pred, true_label) in enumerate(zip(mlp_predictions, charm_predictions, test_df['income'])):
        charm_pred = ' ' + charm_pred
        if (mlp_pred == charm_pred):
            count += 1
        if (charm_pred == true_label):
            count2 += 1
        if (mlp_pred == true_label):
            count3 += 1

    print(f"CHARM Accuracy: {count2/len(X_test_encoded):.4f}")
    print(f"Fidelity: {count/len(X_test_encoded):.4f}")
    print(f"MLP accuracy: {count3/len(X_test_encoded):.4f}")

def save_closed_itemsets(closed_itemsets, closed_itemsets_file):
    with open(closed_itemsets_file):
        joblib.dump(closed_itemsets, closed_itemsets_file)
        print("Closed frequent itemsets saved successfully.")

def load_closed_itemsets(closed_itemsets_file):
    if os.path.exists(closed_itemsets_file):
        print("Loading precomputed closed frequent itemsets...")
        return joblib.load(closed_itemsets_file)
    return None

        
def main(argv):
    train_df = pd.read_csv('Adult-real.csv', header=None, names=attributes) 
    train_df['income'] = train_df['income'].str.replace('.', '', regex=False)

    transaction_data = train_df.values.tolist()
    class_transactions = read_transactions(transaction_data)

    test_df = pd.read_csv('Adult-test.csv', header=None, names=attributes)
    test_df['income'] = test_df['income'].str.replace('.', '', regex=False)
    
    mlp_model = joblib.load('mlp_model.pkl')
    enc = joblib.load('encoder.pkl')

    percentages = [0.1]

    for percentage in percentages:
        class_itemset = defaultdict(list)
        i = 0
        
        for class_label, transactions in class_transactions.items():
            min_support = get_min_support(len(transactions), percentage)
            closed_itemsets = charm(transactions, min_support)
            
            # Uncomment to save trained model
            # closed_itemsets_file =  "Class_" + str(i) + "_save.pkl"
            # save_closed_itemsets(closed_itemsets, closed_itemsets_file)
            class_itemset[class_label].extend([set(itemset) for itemset in closed_itemsets])
            i +=1

        # Uncomment to read the saved model
        # i = 0
        # for class_label, transactions in class_transactions.items():
        #     closed_itemsets_file = "Class_" + str(i) + "_save.pkl"
        #     closed_itemsets = load_closed_itemsets(closed_itemsets_file)
        #     class_itemset[class_label].extend([set(itemset) for itemset in closed_itemsets])
        #     i += 1
        
        support_scores = calculate_support_score(closed_itemsets, class_transactions)
        
        # Uncomment to print class-wise explanations
        # class_explanation, class_score = find_class_explainable_itemset(class_itemset, support_scores)
        # for class_label in class_explanation:
        #     print("Class: ", class_label)
        #     print("Explanation for class: ", class_explanation[class_label])
        #     print("Explanation confidence score: ", class_score[class_label])
        
        compare_mlp_charm(test_df, mlp_model, enc, class_itemset, support_scores)


if __name__ == "__main__":
    main(sys.argv[1:])