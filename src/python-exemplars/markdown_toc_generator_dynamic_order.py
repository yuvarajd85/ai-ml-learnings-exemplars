'''
Created on 2/1/2025 at 1:42 AM
By yuvaraj
Module Name markdown_toc_generator_dynamic_order
'''
from functools import reduce

from dotenv import load_dotenv

load_dotenv()

def main():
    paths = ["../../docs/markdowns/toc-test-1.md","../../docs/markdowns/toc-test.md"]
    [generate_toc(file_path) for file_path in paths]

def generate_toc(file_path:str):
    tab_char = '  '
    toc = f"""## Table Of Contents
"""
    header_list = []
    prev_header_count = 0

    with open(file_path,encoding='utf-8') as f:
        for line in f:
            if str(line).startswith("#"):
                header = line.replace("\n", "").replace("\t", "").strip()
                header_count = header.count("#") if header.__contains__("#") else -1
                header_content = ((header.replace("#", "").replace("*", "").strip().replace(".", "")
                                   .replace("(", "").replace(")", "").replace("?", ""))
                                  .replace("&", ""))
                if len(header_list) == 0:
                    header_list.append(f"- [{header_content}](#{header_content.replace(' ','-').lower()})\n")
                else:
                    prev_tab_count = header_list[len(header_list)-1].count(tab_char)
                    if header_count == prev_header_count:
                        header_list.append(f"{tab_char * prev_tab_count}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n")
                    elif header_count > prev_header_count:
                        header_list.append(f"{tab_char * (prev_tab_count + 1)}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n")
                    elif header_count < prev_header_count:
                        header_list.append(f"{tab_char * (prev_tab_count - 1)}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n")
                prev_header_count = header_count

    toc += reduce(lambda a,b: a+b, header_list)

    print(toc)

if __name__ == '__main__':
    main()
