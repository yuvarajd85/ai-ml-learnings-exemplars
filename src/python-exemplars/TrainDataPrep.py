'''
Created on 2/25/2025 at 11:33 AM
By yuvaraj
Module Name: TrainDataPrep
'''
import json
import random


def generate_simple_spec():
    # Choose a random table name from a list.
    tables = ["db.table_a", "db.table_b", "db.table_c"]
    table = random.choice(tables)
    cols = ["name", "id"]
    # Randomly pick 3 unique IDs between 1 and 100.
    id_values = random.sample(range(1, 101), 3)
    id_str = ",".join(map(str, id_values))

    # Build the spec.
    spec = {
        "table": table,
        "cols": cols,
        "filters": [
            {"column": "id", "operator": "IN", "value": id_str}
        ]
    }
    # Create a dummy SQL output.
    sql = (
        f"SELECT \n    {', '.join(cols)}\nFROM {table}\n"
        f"WHERE id IN ({', '.join(map(str, id_values))});"
    )
    return {"instruction": "Generate SQL query using the following spec:", "input": json.dumps(spec), "output": sql}


def generate_single_join_spec():
    # Base table is chosen from a list.
    base_tables = ["db.table_emp", "db.table_staff"]
    base_table = random.choice(base_tables)
    base_cols = ["name", "id"]
    # Random base filter on id.
    id_values = random.sample(range(1, 101), 3)
    id_str = ",".join(map(str, id_values))
    base_spec = {
        "table": base_table,
        "cols": base_cols,
        "filters": [
            {"column": "id", "operator": "IN", "value": id_str}
        ]
    }

    # Define a join table spec.
    join_tables_list = ["db.table_dept", "db.table_deptinfo"]
    join_table = random.choice(join_tables_list)
    join_cols = ["id", "dept"]
    dept_list = ["assembly", "painting", "sales", "hr"]
    # Randomly pick 2 departments.
    join_depts = random.sample(dept_list, 2)
    join_dept_str = ",".join(join_depts)

    join_spec = {
        "table": join_table,
        "cols": join_cols,
        "how": "inner",
        "leftTable": base_table,
        "leftKeys": ["id"],
        "rightKeys": ["id"],
        "filters": [
            {"column": "dept", "operator": "IN", "value": join_dept_str}
        ]
    }

    spec = {
        "base_table": base_spec,
        "join_tables": [join_spec]
    }

    # Build a dummy SQL query output.
    base_alias = base_table.split(".")[-1]
    join_alias = join_table.split(".")[-1]
    join_dept_values = ", ".join(f"'{d}'" for d in join_depts)
    sql = (
        f"SELECT \n    {base_alias}.name, {base_alias}.id, {join_alias}.dept\n"
        f"FROM {base_table} AS {base_alias}\n"
        f"INNER JOIN {join_table} AS {join_alias}\n"
        f"    ON {base_alias}.id = {join_alias}.id\n"
        f"WHERE {base_alias}.id IN ({', '.join(map(str, id_values))})\n"
        f"  AND {join_alias}.dept IN ({join_dept_values});"
        )
    return {"instruction": "Generate SQL query using the following spec:", "input": json.dumps(spec), "output": sql}


def generate_multiple_join_spec():
    # Define a more complex spec with multiple joins.
    base_table = "db.table_main"
    base_cols = ["col1", "col2", "col3"]
    # Create a random filter for the base table.
    col1_values = random.sample(range(1, 101), 3)
    col1_str = ",".join(map(str, col1_values))
    base_spec = {
        "table": base_table,
        "cols": base_cols,
        "filters": [
            {"column": "col1", "operator": "IN", "value": col1_str}
        ]
    }

    # First join table.
    join1_table = "db.table_join1"
    join1_cols = ["colA", "colB"]
    # Randomly select a filter value for join1.
    join1_filter_value = random.choice(["X", "Y", "Z"])
    join1_spec = {
        "table": join1_table,
        "cols": join1_cols,
        "how": "inner",
        "leftTable": base_table,
        "leftKeys": ["col2"],
        "rightKeys": ["colA"],
        "filters": [
            {"column": "colA", "operator": "=", "value": join1_filter_value}
        ]
    }

    # Second join table.
    join2_table = "db.table_join2"
    join2_cols = ["colM", "colN"]
    join2_filter_values = random.sample(["a", "b", "c", "d"], 2)
    join2_filter_str = ",".join(join2_filter_values)
    join2_spec = {
        "table": join2_table,
        "cols": join2_cols,
        "how": "left",
        "leftTable": join1_table,
        "leftKeys": ["colB"],
        "rightKeys": ["colM"],
        "filters": [
            {"column": "colM", "operator": "IN", "value": join2_filter_str}
        ]
    }

    spec = {
        "base_table": base_spec,
        "join_tables": [join1_spec, join2_spec]
    }

    # Build a dummy SQL query output.
    m_alias = "m"  # alias for base table
    j1_alias = "j1"
    j2_alias = "j2"
    join2_values_str = ", ".join(f"'{val}'" for val in join2_filter_values)
    sql = (
        f"SELECT \n    m.col1, m.col2, m.col3, {j1_alias}.colB, {j2_alias}.colM, {j2_alias}.colN\n"
        f"FROM {base_table} AS {m_alias}\n"
        f"INNER JOIN {join1_table} AS {j1_alias}\n"
        f"    ON m.col2 = {j1_alias}.colA\n"
        f"LEFT JOIN {join2_table} AS {j2_alias}\n"
        f"    ON {j1_alias}.colB = {j2_alias}.colM\n"
        f"WHERE m.col1 IN ({', '.join(map(str, col1_values))})\n"
        f"  AND {j1_alias}.colA = '{join1_filter_value}'\n"
        f"  AND {j2_alias}.colM IN ({join2_values_str});"
        )
    return {"instruction": "Generate SQL query using the following spec:", "input": json.dumps(spec), "output": sql}


# Total number of samples
total_samples = 1000
samples = []

# Use different spec types with a mix:
# ~40% simple, ~30% single join, ~30% multiple join.
for _ in range(total_samples):
    rand_val = random.random()
    if rand_val < 0.4:
        sample = generate_simple_spec()
    elif rand_val < 0.7:
        sample = generate_single_join_spec()
    else:
        sample = generate_multiple_join_spec()
    samples.append(sample)

# Write samples in JSON Lines format to a file.
with open("training_data.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("Generated 1000 training samples in training_data.jsonl")

