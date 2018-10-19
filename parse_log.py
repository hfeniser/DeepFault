import ast
from pylatex import Document, Section, Subsection, Tabular, MultiColumn, MultiRow


def parse_logfile(logfile):

    with open(logfile,'r') as logfile:
            content = logfile.readlines()

    content = [line.strip() for line in content]

    scores     = []
    parameters = []

    for line in content:
        if line[:8] == '{\'mutate':
            param_dict = ast.literal_eval(line)
            parameters.append(param_dict)

        if line[:5] == 'Model':


            ##Make corrections on the string so that it has the form of a
            ##dictionary
            #[left, right] = result.split('Layer')
            #result = left + 'Layer:' + right
            #[left, right] = result.split(' Score')
            #result = left + ', Score' + right

            elements = line.split(': ')
            score = elements[-1]
            scores.append(ast.literal_eval(score))


            '''
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
            '''
            
    return parameters, scores

def generate_latex(results_list):
    doc = Document("LatexTables")
    section = Section('LatexTables')
    subsection = Subsection('Tables')

    ############################
    #######EVALUATION 1#########
    ############################
    table1 = Tabular('|c|c|c|c|c|c|c|c|c|c|')
    table1.add_hline()
    table1.add_row('percentile',(MultiColumn(3, align='|c|',
                                             data='Model_A')),(MultiColumn(3,
                                                                          align='|c|',
                                                                          data='Model_B')),(MultiColumn(3,
                                                                                                       align='|c|',
                                                                                                       data='Model_C')),)
    table1.add_hline()
    table1.add_row(('','T','O','R','T','O','R','T','O','R'))
    table1.add_hline()
    table1.add_row((90, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    table1.add_hline()
    table1.add_row((95, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    table1.add_hline()
    row_cells = (99, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    #row_cells = ('9', MultiColumn(3, align='|c|', data='Multicolumn not on left'))
    table1.add_row(row_cells)
    table1.add_hline()


    ############################
    #######EVALUATION 2#########
    ############################
    table2 = Tabular('|c|c|c|c|c|c|c|c|c|c|')
    table2.add_hline()
    table2.add_row('distance',(MultiColumn(3, align='|c|',
                                             data='Model_A')),(MultiColumn(3,
                                                                          align='|c|',
                                                                          data='Model_B')),(MultiColumn(3,
                                                                                                       align='|c|',
                                                                                                       data='Model_C')),)

    table2.add_hline()
    table2.add_row(('','T','O','R','T','O','R','T','O','R'))
    table2.add_hline()
    table2.add_row((0.2, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    table2.add_hline()
    table2.add_row((0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    table2.add_hline()
    table2.add_row((0.05, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    table2.add_hline()
    row_cells = (0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    #row_cells = ('9', MultiColumn(3, align='|c|', data='Multicolumn not on left'))
    table2.add_row(row_cells)
    table2.add_hline()



    subsection.append(table1)
    subsection.append(table2)

    section.append(subsection)

    doc.append(section)
    doc.generate_pdf(clean_tex=False)


def calculate_value(parameters, scores, expression):
    acc = 0
    cnt = 0
    loss = 0
    for i in range(len(parameters)):

        param_dict = parameters[i]
        if eval(expression):
            #print param_dict
            cnt += 1
            score = scores[i]
            acc += float(score[1])
            loss += float(score[0])

    #print loss
    #print acc
    #print cnt
    average_loss = loss/cnt
    average_acc  = acc/cnt
    print 'Average loss: ' + str(average_loss)
    print 'Average acc: ' + str(average_acc)
    print 'Number of experiments: ' + str(cnt)
    print '-------'

    return 50


if __name__ == '__main__':
    parameters, scores = parse_logfile('laughing-waffle/experiment/logfile.log')
#    results_list = parse_logfile('merged.log')

    print len(parameters)
    print len(scores)
    percentiles = [90, 95, 99]
    distances   = [0.2, 0.1, 0.05, 0.01]
    approaches  = ['ochiai', 'random', 'opposite']
    activations = ['relu', 'leaky_relu']
    classes     = [0,1,3,4,6]
    susp_num    = [1,2,3,5,10]
    model_names = ['mnist_test_model_5_30', 'mnist_test_model_8_20']

    ##########EXP1#############
    distance = distances[1]
    label = classes[0]
    for mn in model_names:
        #for label in classes:
        for sn in susp_num:
            for approach in approaches:
                print('Model: ' + mn + ' approach: ' + approach + ' sn: ' +\
                      str(sn) + ' label: all') #+ str(label))
                expression = '\'' + mn + '\' in param_dict[\'model\'] and param_dict[\'approach\']==\'' +\
                        approach + '\' and param_dict[\'suspicious_num\']==\'' + str(sn) + '\'' + \
                        ' and param_dict[\'distance\'] ==\'' + str(distance) + \
                        '\'' #and param_dict[\'class\']==\'' + str(label) +'\''
                calculate_value(parameters, scores, expression)

    exit()
    generate_latex(results_list)
