'''
Created on 1/15/2025 at 8:58 PM
By yuvaraj
Module Name markdown_toc_generator_heirarchical_order
'''
from dotenv import load_dotenv

load_dotenv()


def main():
    toc = f"""## Table Of Contents
"""
    tab_char = '  '

    with open(f"../../docs/markdowns/toc-test.md",'r',encoding='utf-8') as f:
        for line in f:
            if (str(line).startswith("#")):
                header = line.replace("\n","").replace("\t","").strip()
                header_count = header.count("#") if header.__contains__("#") else -1
                header_content = ((header.replace("#", "").replace("*", "").strip().replace(".", "")
                                   .replace("(", "").replace(")", "").replace("?", ""))
                                  .replace("&", ""))
                match header_count:
                    case 1:
                        toc += f"- [{header_content}](#{header_content.replace(' ', '-').lower()})\n"
                    case 2:
                        toc += f"{tab_char * (header_count)}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n"
                    case 3:
                        toc += f"{tab_char * (header_count)}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n"
                    case 4:
                        toc += f"{tab_char * (header_count)}- [{header_content}](#{header_content.replace(' ', '-').lower()})\n"


    print(toc)


if __name__ == '__main__':
    main()
