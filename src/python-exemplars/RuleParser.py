'''
Created on 7/2/2025 at 1:27 AM
By yuvaraj
Module Name: RuleParser
'''
import re
from typing import Dict, Union
from dotenv import load_dotenv

load_dotenv()



def parse_dqdl_rule(rule: str) -> Dict[str, Union[str, Dict]]:
    # Trim whitespace
    rule = rule.strip()

    # Regular expression to split RuleType and body
    match = re.match(r'^(\w+)\s+"([^"]*)"\s*(.*)$', rule)

    if not match:
        # Possibly a rule type like CustomSql without a column name
        match_alt = re.match(r'^(\w+)\s+(.*)$', rule)
        if not match_alt:
            return {"Error": "Invalid Rule Format"}
        rule_type = match_alt.group(1)
        column_name = ""
        expression = match_alt.group(2).strip()
    else:
        rule_type = match.group(1)
        column_name = match.group(2)
        expression = match.group(3).strip()

    # Parse expression into components
    expression_details = parse_expression(expression)

    return {
        "RuleType": rule_type,
        "ColumnName": column_name,
        "Expression": expression,
        "ExpressionDetails": expression_details
    }


def parse_expression(expression: str) -> Dict[str, Union[str, list]]:
    # Normalize and identify operators
    patterns = {
        'between': r'between\s+([\d\.]+)\s+and\s+([\d\.]+)',
        'equals': r'^=\s*([\d\.]+)$',
        'greater_than': r'^>\s*([\d\.]+)$',
        'less_than': r'^<\s*([\d\.]+)$',
        'in': r'in\s*\((.*?)\)',
        'like': r'like\s+"([^"]+)"',
    }

    for op, pat in patterns.items():
        match = re.search(pat, expression, re.IGNORECASE)
        if match:
            if op == 'between':
                return {"Operator": "between", "Values": [float(match.group(1)), float(match.group(2))]}
            elif op == 'in':
                values = [v.strip() for v in match.group(1).split(',')]
                return {"Operator": "in", "Values": values}
            elif op == 'like':
                return {"Operator": "like", "Pattern": match.group(1)}
            else:
                return {"Operator": op.replace('_', ' '), "Value": match.group(1)}

    return {"Operator": "unknown", "Raw": expression}


def main():
    rules = ['ColumnValues "Customer_ID" between 1 and 2000',
             'ColumnLength "Postal_Code" = 5',
             'CustomSql select count(*) from database.table inner join database.table2 on id1 = id2 between 10 and 20'
             ]

    for rule in rules:
        print(parse_dqdl_rule(rule))


if __name__ == '__main__':
    main()
