import matplotlib.pyplot as plt
import mysklearn.myutils as myutils
import mysklearn.mypytable as mypytable
import numpy as np

def make_bar_graph_with_column(column, title, x_label, y_label, size):
    plt.figure()
    dictionary = myutils.create_dictionary(column)
    plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
    plt.xticks(range(len(dictionary)), list(dictionary.keys()))
    plt.xticks(fontsize=size)
    plt.xticks(rotation = 90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def make_bar_graph_with_xy(x, y, title, x_label, y_label, size):
    plt.figure()
    plt.bar(x, y, align='center')
    plt.xticks(fontsize=size)
    plt.xticks(rotation = 90)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def get_frequencies(col):
    values = []
    counts = []
    ints = []
    counter = 0

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
            ints.append(counter)
            counter = counter + 1
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts, ints

def make_pie_chart_with_comp(attributes, frequency, title):
    # reset figure, make it square
    plt.figure(figsize=(8,8))

    # define x and y values
    xs = attributes
    ys = []
    total = 0

    for item in frequency:
        total += item

    for item in frequency:
        ys.append(item/total)

    # create the bar chart (with pcts)
    
    plt.title(title)
    plt.pie(ys, labels=xs, autopct="%1.1f%%", textprops={'fontsize':8})
    plt.show()

def make_histogram(table, attribute, title, x_label, y_label):
    # reset figure
    plt.figure()

    #xs = table.get_column(attribute)
    xs = attribute

    # create the bar chart (alpha is transparency, b is blue)
    plt.hist(xs, bins=20, alpha=0.75, color="b")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label, size = 8)
    plt.xticks(rotation = 45)
    # turn on the background grid
    plt.grid(True)
    plt.show()

def make_scatter_plot(table, x, y, title, x_label, y_label):
    # reset figure
    plt.figure()

    # create xs and ys
    xs = table.get_column(x)
    ys = table.get_column(y)

    # create the scatter plot
    plt.plot(xs, ys, "b.")

    # make axis a bit longer and wider
    plt.xlim(0, int(max(xs) * 1.10))
    plt.ylim(0, int(max(ys) * 1.10))

    # get linear regression 
    m, b, r, c = myutils.get_regression_vals(xs, ys)
    y_lst = []
    y = 0

    for item in xs:
        y = m * item + b
        y_lst.append(y)

    # place a text box left corner
    textstr = "r = {} \ncov = {}".format(r, c)
    plt.text(1,(max(ys) - .1 * max(ys)), textstr)

    plt.plot(xs, y_lst)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def make_scatter_plot_multiple(x1, x2, title, x_label, y_label):
    # reset figure
    plt.figure()
    
    ax = plt.gca()

    ax.scatter(x1, x2, color="b")
    plt.xlim(0, int(max(x2) * 1.10))
    plt.ylim(0, int(max(x2) * 1.10))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def make_multiple_freq_diagram(a_lst_1, a_lst_2, title, x_label, y_label, attribute1, attribute2, cutoffs):
    # create the figure
    plt.figure()

    a1_dict = myutils.create_dictionary(a_lst_1)
    a2_dict = myutils.create_dictionary(a_lst_2)

    freq_1 = myutils.compute_bin_frequencies(a_lst_1, cutoffs)
    freq_2 = myutils.compute_bin_frequencies(a_lst_2, cutoffs)

    ax = plt.gca()

    cutoffs_short = []
    for i in range(len(cutoffs) - 1):
        cutoffs_short.append(cutoffs[i])
    
    cutoffs_plus = []
    for item in cutoffs_short:
        cutoffs_plus.append(item + .3)

    # spacing can be a tricky
    r1 = ax.bar(cutoffs_short, freq_1, 0.3, color="r")
    r2 = ax.bar(cutoffs_plus, freq_2, 0.3, color="b")

    ax.set_xticks(cutoffs_short)
    
    # set x value labels
    labels = []
    for i in cutoffs_short:
        labels.append(i)
    ax.set_xticklabels(labels)
    

    # create a legend
    ax.legend((r1[0], r2[0]), (attribute1, attribute2), loc=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def make_box_and_whisker(freq_lst, name_lst, title, x_label, y_label, size):
    # create the figure
    plt.figure()     

    # create boxplot with two distributions
    plt.boxplot(freq_lst)
    # set x-axis value names
    tick_marks = []
    for i in range(len(name_lst)):
        tick_marks.append(i + 1)
    plt.xticks(tick_marks, name_lst)
    plt.xticks(rotation = 45)
    plt.xticks(fontsize=size)
    plt.title(title)
    plt.show()