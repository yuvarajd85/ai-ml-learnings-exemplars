from typing import List, Tuple


def main():
    print_table(4,3, None, None)
    col_names = ["Name", "Id", "Dept", "Desig"]
    data = [("Yuvi","044633", "GIFS","TL-III"),("Dharani","056313", "PI","TL-I"),("Amora","164216", "INTL","DEV-I")]
    print_table(2,1,data,col_names)

def print_table(col:int, row:int, data: List[Tuple], col_name: List):
    if col_name:
        col = len(col_name)
    if data:
        row = len(data)

    table_str = ""
    hdr = ""
    for i in range(0, (col + 1)):
        hdr+= "|"

        if not (i == col):
            if col_name:
                hdr += col_name[i]
            else:
                hdr += f"Col_{i+1}"
    table_str += f"{hdr}\n"

    sep = ""
    for i in range(0, (col + 1)):
        sep += "|"
        if not (i == col):
            if i == 0:
                sep += ":---"
            elif (i==(col - 1)):
                sep += "---:"
            else:
                sep += ":---:"
    table_str += f"{sep}\n"

    for r in range(0, (row)):
        row_val = ''
        for i in range(0, (col + 1)):
            row_val+="|"
            if not (i == col):
                if data:
                    row_val += data[r][i]
                else:
                    row_val += f"value_{i+1}"
        table_str += f"{row_val}\n"

    print(table_str)


if __name__ == '__main__':
    main()