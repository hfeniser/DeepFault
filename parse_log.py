import ast
from pylatex import Document, Section, Subsection, Tabular, MultiColumn, MultiRow

with open('experiment/logfile_2.log','r') as logfile:
        content = logfile.readlines()

content = [line.strip() for line in content]

results_list =[]
for line in content:
    if line[:5] == 'Model':

        result = line

        ##Make corrections on the string so that it has the form of a
        ##dictionary
        [left, right] = result.split('Layer')
        result = left + 'Layer:' + right
        [left, right] = result.split(' Score')
        result = left + ', Score' + right

        elements = result.split(',')
        if not 'No Suspicious' in elements[-1]:
            list_elem = elements[-2] + ', ' + elements[-1]
            elements[-2] = list_elem
            elements = elements[:-1]

        result = ''
        for elem in elements:
            [key, value] = elem.split(':')
            key = key.strip()
            value = value.strip()
            result +=  '\'' + key + '\':'
            value = '\'' + value + '\''
            result += value + ','

        ##Add brackets to make it dictionary
        result = '{' + result + '}'
        result_dict = ast.literal_eval(result)
        results_list.append(result_dict)

print results_list[0]


doc = Document("LatexTables")
section = Section('LatexTables')
subsection = Subsection('Tables')


table2 = Tabular('|c|c|c|')
table2.add_hline()
table2.add_row((MultiRow(3, data='Multirow'), 1, 2))
table2.add_hline(2, 3)
table2.add_row(('', 3, 4))
table2.add_hline(2, 3)
table2.add_row(('', 5, 6))
table2.add_hline()
table2.add_row((MultiRow(3, data='Multirow2'), '', ''))
table2.add_empty_row()
table2.add_empty_row()
table2.add_hline()


subsection.append(table2)
section.append(subsection)

doc.append(section)

