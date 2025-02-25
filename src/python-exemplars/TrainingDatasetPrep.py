'''
Created on 2/25/2025 at 11:01 AM
By yuvaraj
Module Name: TrainingDatasetPrep
'''
from dotenv import load_dotenv
import json

load_dotenv()


def main():
    input = {
        "base_table": {
            "table": "db.table_emp",
            "cols": ["name", "id"],
            "filters": [{"column": "id", "operator": "IN", "value": "1,23,5"}]
        },
        "join_tables": [
            {
                "table": "db.table_emp",
                "cols": ["id", "dept", "division"],
                "how": "inner",
                "leftTable": "db.table_dept",
                "leftKeys": ["id"],
                "rightKeys": ["id"],
                "filters": [{"column": "dept", "operator": "IN", "value": "assembly,painting"}]
            },
            {
                "table": "db.table_emp",
                "cols": ["id", "service_start_date", "service_end_date", "service_dept"],
                "how": "inner",
                "leftTable": "db.table_service",
                "leftKeys": ["id"],
                "rightKeys": ["id"],
                "filters": [{"column": "service_start_date", "operator": "<", "value": ":current-date"}]
            }
        ]
    }

    output = """
    SELECT 
        e.name, 
        e.id,
        d.id      AS dept_id, 
        d.dept, 
        d.division,
        s.id      AS service_id,
        s.service_start_date, 
        s.service_end_date, 
        s.service_dept
    FROM db.table_emp AS e
    INNER JOIN db.table_dept AS d
        ON e.id = d.id
    INNER JOIN db.table_service AS s
        ON e.id = s.id
    WHERE e.id IN (1, 23, 5)
      AND d.dept IN ('assembly', 'painting')
      AND s.service_start_date < :current_date;
    """

    instruction = "Generate the sql using the following spec: "

    training_data = {"instruction": instruction,"input": json.dumps(input), "output" : json.dumps(output) }

    print(training_data)


if __name__ == '__main__':
    main()
