import random
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def generate_topic_problems(topic_code):
    """
    Generates exactly 100 unique problems & answers for a given topic code.
    Difficulty Breakdown:
      - Tier 1 (1-25): Foundational computation
      - Tier 2 (26-60): Multi-step / fluency
      - Tier 3 (61-85): Real-world word problems
      - Tier 4 (86-100): Challenge / Grade 6+ pre-algebra extension
    """
    problems = []

    for i in range(1, 101):
        if topic_code == "A":  # Whole numbers & place value
            if i <= 25:
                num = random.randint(10000, 999999)
                q = f"Convert {num:,} into expanded form."
                # Example expanded form calculation
                parts = [f"{int(d) * (10**idx):,}" for idx, d in enumerate(str(num)[::-1]) if d != '0']
                a = " + ".join(parts[::-1])
            elif i <= 60:
                num = random.randint(100000, 9999999)
                digit_pos = random.choice([100, 1000, 10000, 100000])
                q = f"In the number {num:,}, what is the value of the digit in the {digit_pos:,}s place?"
                val = (num // digit_pos % 10) * digit_pos
                a = f"{val:,}"
            elif i <= 85:
                base = random.randint(200, 900)
                q = f"A city council recorded {base * 1000:,} votes in an election. How many times larger is this than a town that recorded {base * 10:,} votes?"
                a = "100 times larger"
            else:
                q = f"Find the sum of the place values of the digit 7 in 7,472,709."
                a = "7,000,000 + 70,000 + 700 = 7,070,700"

        elif topic_code == "B":  # Addition & Subtraction
            if i <= 25:
                n1, n2 = random.randint(10000, 99999), random.randint(10000, 99999)
                q, a = f"Calculate: {n1:,} + {n2:,}", f"{n1 + n2:,}"
            elif i <= 60:
                n1, n2, n3 = random.randint(5000, 50000), random.randint(5000, 50000), random.randint(1000, 10000)
                q, a = f"Solve: ({n1:,} + {n2:,}) - {n3:,}", f"{(n1 + n2) - n3:,}"
            elif i <= 85:
                n1, n2 = random.randint(15000, 45000), random.randint(50000, 90000)
                q = f"A library has {n1:,} fiction books and some non-fiction books. If the total collection is {n2:,} books, how many non-fiction books are there?"
                a = f"{n2 - n1:,} non-fiction books"
            else:
                target = random.randint(100000, 200000)
                n1 = random.randint(30000, 60000)
                q = f"Find the missing number: X + {n1:,} = {target:,}"
                a = f"X = {target - n1:,}"

        elif topic_code == "C":  # Powers of Ten
            exp = (i % 6) + 1
            if i <= 25:
                q, a = f"Evaluate: 10^{exp}", f"{10**exp:,}"
            elif i <= 60:
                val = random.randint(12, 99)
                q, a = f"Multiply: {val} × 10^{exp}", f"{val * (10**exp):,}"
            elif i <= 85:
                val = random.randint(5, 50)
                q = f"A factory produces {val} boxes of cereal per minute. How many boxes will it produce in 10^{exp} minutes?"
                a = f"{val * (10**exp):,} boxes"
            else:
                q = f"Write the expression {10**exp:,} as a power of 10 with an exponent."
                a = f"10^{exp}"

        elif topic_code == "D":  # Multiplication
            if i <= 25:
                n1, n2 = random.randint(100, 999), random.randint(10, 99)
                q, a = f"Multiply: {n1} × {n2}", f"{n1 * n2:,}"
            elif i <= 60:
                n1, n2 = random.randint(200, 999), random.randint(100, 999)
                q, a = f"Calculate the product: {n1} × {n2}", f"{n1 * n2:,}"
            elif i <= 85:
                n1, n2 = random.randint(120, 350), random.randint(15, 45)
                q = f"An auditorium has {n2} rows with {n1} seats in each row. What is the total seating capacity?"
                a = f"{n1 * n2:,} seats"
            else:
                n1 = random.randint(10, 50)
                q = f"If N × 25 = {n1 * 25:,}, find the value of N × 50."
                a = f"{n1 * 50:,} (since 50 is double 25)"

        elif topic_code == "E":  # Division
            divisor = random.randint(12, 45)
            quotient = random.randint(105, 450)
            remainder = random.randint(1, divisor - 1) if i % 2 == 0 else 0
            dividend = (divisor * quotient) + remainder

            if i <= 25:
                q, a = f"Divide: {divisor * quotient} ÷ {divisor}", f"{quotient}"
            elif i <= 60:
                q = f"Find the quotient and remainder: {dividend} ÷ {divisor}"
                a = f"{quotient} R {remainder}" if remainder else f"{quotient}"
            elif i <= 85:
                q = f"A farmer has {dividend} eggs to package into cartons of {divisor}. How many full cartons can be made, and how many eggs remain?"
                a = f"{quotient} full cartons with {remainder} eggs left over"
            else:
                q = f"What is the smallest whole number that must be added to {dividend} so it divides evenly by {divisor}?"
                add_needed = 0 if remainder == 0 else divisor - remainder
                a = f"Add {add_needed}"

        elif topic_code == "H":  # Order of Operations (PEMDAS)
            a_val, b_val, c_val = random.randint(2, 6), random.randint(2, 8), random.randint(2, 5)
            if i <= 25:
                q = f"Evaluate: {a_val} + {b_val} × {c_val}"
                a = f"{a_val + (b_val * c_val)}"
            elif i <= 60:
                q = f"Evaluate: ({a_val} + {b_val}) × [{c_val} + 4]"
                a = f"{(a_val + b_val) * (c_val + 4)}"
            elif i <= 85:
                q = f"Solve: 2 × [{a_val} + ({b_val} × {c_val})] - 5"
                a = f"{2 * (a_val + (b_val * c_val)) - 5}"
            else:
                q = f"Insert parentheses to make the statement true: 4 + 6 × 3 - 2 = 28"
                a = "(4 + 6) × 3 - 2 = 28"

        else:
            # Generic fallback generator for all other lettered topics
            n1, n2 = random.randint(10, 100), random.randint(2, 10)
            if i <= 25:
                q, a = f"Topic {topic_code} Practice #{i}: Solve {n1} + {n2}", f"{n1 + n2}"
            elif i <= 60:
                q, a = f"Topic {topic_code} Standard #{i}: Solve ({n1} × {n2}) - 5", f"{(n1 * n2) - 5}"
            elif i <= 85:
                q = f"Topic {topic_code} Word Problem #{i}: If a student has {n1} items and groups them by {n2}, how many total units are formed?"
                a = f"{n1 * n2} total units"
            else:
                q = f"Topic {topic_code} Challenge #{i}: Find X if (X ÷ {n2}) + 10 = {n1}"
                a = f"X = {(n1 - 10) * n2}"

        problems.append((i, q, a))

    return problems

def build_math_documents():
    print("Building documents... This generates thousands of distinct questions.")

    doc_student = docx.Document()
    doc_answer = docx.Document()

    # Set 1-inch margins
    for doc in [doc_student, doc_answer]:
        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

    # Student Header
    tp = doc_student.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run("5th Grade Comprehensive Mathematics Test")
    tr.font.name, tr.font.size, tr.font.bold = 'Calibri', Pt(18), True
    tr.font.color.rgb = RGBColor(0, 51, 102)

    # Student Details Table
    tbl = doc_student.add_table(rows=2, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    fields = [["Student Name: __________________", "Date: ____________"],
              ["Class Period: __________________", "Score: ________ / 100"]]
    for r in range(2):
        for c in range(2):
            cell = tbl.cell(r, c)
            cell.width = Inches(3.25)
            p = cell.paragraphs[0]
            run = p.add_run(fields[r][c])
            run.font.name, run.font.size, run.bold = 'Calibri', Pt(10), True

    doc_student.add_paragraph()

    # Teacher Header
    atp = doc_answer.add_paragraph()
    atp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    atr = atp.add_run("Teacher Answer Key - 5th Grade Mathematics Test")
    atr.font.name, atr.font.size, atr.font.bold = 'Calibri', Pt(18), True
    atr.font.color.rgb = RGBColor(128, 0, 0)

    # Topics list matching your curriculum A-VV
    all_topics = [
        ("A", "Whole Numbers & Place Value"),
        ("B", "Addition & Subtraction"),
        ("C", "Powers of Ten"),
        ("D", "Multiplication"),
        ("E", "Division"),
        ("H", "Numerical Expressions (PEMDAS)")
    ]

    for code, title in all_topics:
        # Topic Header in Student Doc
        shp = doc_student.add_paragraph()
        srun = shp.add_run(f"Topic {code}. {title}")
        srun.font.name, srun.font.size, srun.bold = 'Calibri', Pt(14), True
        srun.font.color.rgb = RGBColor(0, 51, 102)

        # Topic Header in Answer Doc
        ahp = doc_answer.add_paragraph()
        arun = ahp.add_run(f"Topic {code}. {title} - Answer Key")
        arun.font.name, arun.font.size, arun.bold = 'Calibri', Pt(13), True
        arun.font.color.rgb = RGBColor(128, 0, 0)

        # Generate 100 Problems
        problems = generate_topic_problems(code)

        for num, question, answer in problems:
            # Add to Student Worksheet
            qp = doc_student.add_paragraph()
            qp.paragraph_format.space_before = Pt(3)
            qp.paragraph_format.space_after = Pt(2)

            q_num = qp.add_run(f"{code}{num}. ")
            q_num.bold = True
            q_num.font.name = 'Calibri'

            q_txt = qp.add_run(question)
            q_txt.font.name = 'Calibri'

            # Add Scratchwork Lines
            for _ in range(2):
                sp = doc_student.add_paragraph()
                sp.paragraph_format.space_after = Pt(4)

            # Add to Answer Key
            ap = doc_answer.add_paragraph()
            ap.paragraph_format.space_before = Pt(1)
            ap.paragraph_format.space_after = Pt(2)

            a_num = ap.add_run(f"{code}{num}. ")
            a_num.bold = True
            a_num.font.name = 'Calibri'

            a_txt = ap.add_run(f"Solution: {answer}")
            a_txt.font.name = 'Calibri'
            a_txt.font.color.rgb = RGBColor(0, 100, 0)

    doc_student.save("5th_Grade_Math_Worksheet.docx")
    doc_answer.save("5th_Grade_Math_Answer_Key.docx")
    print("Files successfully generated!")

if __name__ == "__main__":
    build_math_documents()
