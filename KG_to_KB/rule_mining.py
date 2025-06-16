from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
def sub_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    print(len(contents))
    result = []
    criminal_keyword = ["过失致人死亡", "故意伤害", "危害公共安全", "贩卖毒品", "交通肇事", "故意杀人", "危险驾驶", "寻衅滋事"]
    doc = open('sub_KG.txt', 'w', encoding='utf-8')
    for content in contents:
        for criminal in criminal_keyword:
            if criminal in content:
                result.append(content)
                print(content, file=doc)
    print(len(result))

def parse_rule(rule_str):
    pattern = r"frozenset\(\{(.+?)\}\) -> frozenset\(\{(.+?)\}\) \(Confidence: ([\d.]+)\)"
    match = re.match(pattern, rule_str)
    if match:
        premise = frozenset({match.group(1).strip()})
        conclusion = frozenset({match.group(2).strip()})
        confidence = float(match.group(3))
        return (premise, conclusion, confidence)
    return None
def detect_and_remove_conflicts():
    with open('initial_KB.txt', 'r', encoding='utf-8') as f:
        rules = f.read().splitlines()
    #print(rules)
    parsed_rules = [parse_rule(rule_str) for rule_str in rules]
    rules = [rule for rule in parsed_rules if rule is not None]
    to_remove = set()
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            premise_i, conclusion_i, confidence_i = rules[i]
            premise_j, conclusion_j, confidence_j = rules[j]
            #if premise_i == premise_j or conclusion_i == conclusion_j:
            if conclusion_i == conclusion_j:
                to_remove.add(i)
                to_remove.add(j)

    return [rule for idx, rule in enumerate(rules) if idx not in to_remove]

def rule_min(file_path):
    # 三元组数据
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    transactions = []
    for content in contents:
        traiad_list = content.split('\t')
        if len(traiad_list) == 4:
            transactions.append({re.sub('\n',"",traiad_list[0]), re.sub('\n',"",traiad_list[1]), re.sub('\n',"",traiad_list[3])})
    # transactions = [
    #     {'领导干部', '权力观', '加强对党员干部特别是领导干部的群众观、权力观教育。'},
    #     {'独立性', '法律地位', '具体表现为SPV法律地位的独立性、业务范围的限制性、受让应收帐款的真实性、应收帐款价值的不确定性以及信用级别的增强性。'},
    #     # 添加其他三元组...
    # ]
    # 将三元组数据转换为适合挖掘的DataFrame格式
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 使用Apriori算法找出频繁项集
    frequent_itemsets = apriori(df, min_support=0.0005, use_colnames=True)

    # 根据频繁项集找出关联规则，计算置信度
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    doc = open('initial_KB.txt', 'w', encoding='utf-8')
    for index, row in rules.iterrows():
        if row['confidence'] > 0.8:
            print(row['antecedents'], '->', row['consequents'], '(Confidence: {:.2f})'.format(row['confidence']), file=doc)
    #冲突规则检测并筛选
    reduced_KB = detect_and_remove_conflicts()
    doc = open('reduced_KB.txt', 'w', encoding='utf-8')
    for reduced in reduced_KB:
        print(reduced, file=doc)

#detect_and_remove_conflicts()
rule_min('sub_KG.txt')

