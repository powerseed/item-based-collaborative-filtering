import pandas as pd
import numpy as np
def positive_and_boundary_region(data,decision_column_name):
    universe_copy = data.copy()
    attrs = universe_copy.drop([decision_column_name],axis = 1).columns.values
    universe_copy = universe_copy.sort_values(by = attrs.tolist())     
    boundary_set = pd.DataFrame(columns = data.columns)
    if universe_copy.shape[0] != 0:
        i = 0
        while i < universe_copy.shape[0]:
            print(i)
            o1 = universe_copy.drop([decision_column_name],axis = 1).iloc[i].values
            mark_a = i
            mark_b = None
            for j in range(i+1, universe_copy.shape[0]):
                o2 = universe_copy.drop([decision_column_name], axis = 1).iloc[j].values
                if not np.array_equal(o1,o2):## o2 is not equal to o1
                    mark_b = j-1 ##mark the last index that equal to o1
                    break
            if mark_a != mark_b:## that is actually have some obejct in the system that have attribute all equal
                if mark_b == None:## what it mean? it means all the way along o2 is euqal to o1
                    decision_num = universe_copy[decision_column_name][mark_a:].unique().shape[0]
                    if decision_num > 1: ##inconsistent
                        for index in range(mark_a,universe_copy.shape[0]):
                            boundary_set.append(universe_copy.iloc[index])
                    else:## consistent
                        pass##do nothing, it is in the positive region
                    break
                else:
                    decision_num = universe_copy[decision_column_name][mark_a:mark_b+1].unique().shape[0]
                    if decision_num > 1: ##inconsistent
                        for index in range(mark_a,mark_b+1):
                            boundary_set.append(universe_copy.iloc[index])
                    else:## consistent
                        pass##do nothing, it is in the positive region
                    i = mark_b + 1
            else: ## no object is equal to o1
                i = i + 1
    positive_set = exclude(universe_copy,boundary_set)
    return positive_set,boundary_set
                                              
def getCandidate(g):
    candidate = []
    attributes = g.columns.values
    for attr in attributes:
        uniq_values = g[attr].unique()
        for val in uniq_values:
            candidate.append([attr,val])
    return candidate
    
def candidate_exclude_exist_condition(candidate,condition):
    for i in range(len(condition)):
        pairs = condition[i]
        if pairs in candidate:
            candidate.remove(pairs)
    return candidate

def find_cover(condition,data): 
    subset = data.copy()
    for i in range(len(condition)):
        pair = condition[i]
        subset = subset[subset[pair[0]]==pair[1]]
    return subset

def rule_union(rules,data):
    r_union = pd.DataFrame(columns = data.columns)
    if len(rules) > 0:
        r_union = find_cover(rules[0],data)
        for i in range(1,len(rules)):
            r_union = union(r_union,find_cover(rules[i],data))
    return r_union

def contain(a,b):
    b_index = b.index.values
    for i in range(len(b_index)):
        if not b_index[i] in a.index.values:
            return False
    return True

def intersection(a,b):
    inter = pd.DataFrame(columns = a.columns)
    b_index = b.index.values
    a_index = a.index.values
    for i in range(len(b_index)):
        if b_index[i] in a_index:
            inter = inter.append(b.loc[b_index[i]])
    return inter

def union(a,b):
    union_set = a.copy()
    b_index = b.index.values
    a_index = a.index.values
    for i in range(len(b_index)):
        if not b_index[i] in a_index:
            union_set = union_set.append(b.loc[b_index[i]])
    return union_set

def exclude(a,b):
    base_set = pd.DataFrame(columns = a.columns)
    a_index = a.index.values
    b_index = b.index.values
    for i in range(len(a_index)):
        if not a_index[i] in b_index:
            base_set = base_set.append(a.loc[[a_index[i]]])
    return base_set

def difference(conditions,pair):
    copy = conditions.copy()
    copy.remove(pair)
    return copy

def equal_set(a,b):
    if a.shape[0] != b.shape[0]:
        return False
    b_index = np.sort(b.index.values)
    a_index = np.sort(a.index.values)
    for i in range(b_index.shape[0]):
        if a_index[i] != b.index[i]:
            return False
    return True

def select_pair(candidate,dataset,g):
    best_pairs = []
    best_inter_num = -1
    best_cover_num = np.inf##this is the cover number of the best inter
    for pair in candidate:
        coverset = find_cover([pair],dataset)
        inter = intersection(coverset,g)
        if inter.shape[0] > best_inter_num:
            best_inter_num = inter.shape[0]
            best_cover_num = coverset.shape[0]
            best_pairs = []
            best_pairs.append(pair)
        elif inter.shape[0] == best_inter_num:
            if coverset.shape[0] < best_cover_num:
                best_pairs = []
                best_pairs.append(pair)
                best_inter_num = inter.shape[0]
                best_cover_num = coverset.shape[0]
            elif coverset.shape[0] == best_cover_num:
                best_pairs.append(pair)
            else:
                pass
        else:
            pass
    return best_pairs[0]
def rule_stability(condition,decision,universe):
    matching = find_cover(condition,universe).shape[0]
    rule = condition.copy()
    rule.append(decision)
    support = find_cover(rule,universe).shape[0]
    return float(support/matching)

def approximate_rules(universe,all_rules,epsilon):
    for i in range(len(all_rules)):
        rules_decision = all_rules[i]
        decision = rules_decision[1]
        rules = rules_decision[0]
        for i in range(len(rules)):
            copy = rules[i].copy()
            for pair in copy:
                if rule_stability(difference(rules[i],pair),decision,universe) >= 1 - epsilon:
                    rules[i] = difference(rules[i],pair)
    return all_rules
        
def merge_condition(condition,pair):
    if pair in condition:
        pass
    else:
        condition.append(pair)
    return condition

def rule_finding(universe,decision_column_name):
    positive_region, boundary_region = positive_and_boundary_region(universe,decision_column_name)
    decision_values = universe[decision_column_name].unique()
    total_rule = []
    for decision in decision_values:
        print(decision)
        lower_approximate = positive_region[positive_region[decision_column_name] == decision].drop([decision_column_name],axis = 1)
        total_rule.append([lem2(universe.drop([decision_column_name],axis = 1),lower_approximate,decision),[decision_column_name, decision]])
    return total_rule

def lem2(universe,sub_region,decision):
    concept_set = sub_region.copy()
    g = concept_set.copy()
    rules = []
    while g.shape[0] != 0:
        condition = []
        candidate = getCandidate(g)
        while len(condition) == 0 or (not contain(concept_set,find_cover(condition,universe))):
            pair = select_pair(candidate,universe,g)
            condition = merge_condition(condition,pair)
            g = intersection(find_cover([pair],universe),g)
            candidate = getCandidate(g)
            candidate = candidate_exclude_exist_condition(candidate,condition)
            if len(candidate) == 0:
                break
        copy = condition.copy()
        for pair in copy:
            diff_con = difference(condition,pair)
            if contain(concept_set,find_cover(diff_con,universe)):
                condition = diff_con
        rules.append(condition)
        g = exclude(concept_set,rule_union(rules,universe))
    copy = rules.copy()
    for rule in copy:
        diff_rules = difference(rules,rule)
        if equal_set(rule_union(diff_rules,universe), concept_set):
            rules = diff_rules
    return rules
def generate_rule_table(all_rules,columns_name):
    rule_table = pd.DataFrame(columns = columns_name)
    for rules_decision in all_rules:
         decision = rules_decision[1]
         rules = rules_decision[0]
         for rule in rules:
             dic = {decision[0]:decision[1]}
             for pair in rule:
                 dic[pair[0]] = pair[1]
             rule_table = rule_table.append(dic,ignore_index = True)
    rule_table.to_csv('decisionRule.csv')
    return rule_table
def predict(X):
    pass
data = pd.read_csv('trainDataCURADiscretized.csv', index_col = 0)[:200000]
decision_col_name = data.columns.values[-1]
rules = rule_finding(data,decision_col_name)
rules = approximate_rules(data, rules, 0.2)
    
        

