import mysklearn.myutils as myutils
import copy
import csv 
from tabulate import tabulate 
# uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.data[0])

        return N, M 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """

        column = []
        
        try:
            # check if input is string
            if type(col_identifier) == str:
                # get index from name 
                for item in self.column_names:
                    if item == col_identifier:
                        column_num = self.column_names.index(item)
                for row in self.data:
                    # check if user wants to include missing values
                    if include_missing_values:
                        column.append(row[column_num])
                    else:
                        if row[column_num] != "NA":
                            column.append(row[column_num])
            # assumes the argument is an index, if it does not work, the exception will throw
            else:
                for row in self.data:
                    # check if user wants missing values 
                    if include_missing_values:
                        column.append(row[col_identifier])
                    else:
                        if row[col_identifier]:
                            column.append(row[col_identifier])
            
            return column
        # throw an error if it is not a string, int, or is out of range
        except ValueError: 
            print("The datatype for col_identifier is invalid. Please enter a str or int.")
            raise

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[0])):
                try:
                    numeric = float(self.data[i][j])
                    self.data[i][j] = numeric
                    
                except:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        # pass

        clean_table = []
        no_match = 0
        no_match_total = 0
        # only append rows from self to clean_table that are not in rows_to_drop
        for row in self.data:
            no_match_total = 0
            for row2 in rows_to_drop:
                no_match = 0
                for item in range(0, len(self.data[0])):
                    if row[item] != row2[item]:
                        no_match = 1
                if no_match == 1:
                    no_match_total += 1
            if no_match_total == len(rows_to_drop):
                clean_table.append(row)
                        

        self.data = copy.deepcopy(clean_table)
        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        file_data = list(csv.reader(open(filename)))

        header = file_data[0]
        self.column_names = header
        del file_data[0]
        
        data = file_data
        self.data = data
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # use csv writer to write data row by row
        with open(filename, mode = 'w+') as data_file:
            writer = csv.writer(data_file, delimiter=',')
            writer.writerow(self.column_names)
            writer.writerows(self.data)


    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        index_lst = []
        # get indeces of column names
        for name in key_column_names:
            index_lst.append(self.column_names.index(name))

        for i in range(len(self.data)):
            j = i + 1
            for k in range(j, len(self.data)):
                match = 0
                for index in index_lst:
                    if self.data[i][index] == self.data[k][index]:
                        match += 1
                if match == len(key_column_names):
                    if self.data[k] not in duplicates:
                        duplicates.append(self.data[k])
                    match = 0
        return duplicates 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        clean_table = []

        # only add row to clean table if no values are missing
        for row in self.data:
            missing = 0
            for column in row:
                if column == "NA":
                    missing = 1
                if column == "":
                    missing = 1
            if missing == 0:
                clean_table.append(row)
        self.data = copy.deepcopy(clean_table)
        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # get column index
        col_index = self.column_names.index(col_name)

        # get average of column
        sum = 0
        count = 0
        for row in self.data:
            if row[col_index] != "NA":
                sum += row[col_index]
                count += 1
        avg = sum/count
        avg = float(round(avg))

        # replace missing values with average 
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg
            if row[col_index] == "":
                row[col_index] = avg


        return self

    def replace_missing_values_intelligent(self, column_name, key):
        """For columns with continuous data, fill missing values in a column by the average of like values (decided by key)

        Args:
            col_name(str): name of column to fill with the original average (of the column).
            key(str): attribute of choice 
        """
        # get column index
        col_index = self.column_names.index(column_name)
        key_index = self.column_names.index(key)
        sum = 0
        count = 0
        avg = 0

        for row in self.data:
            if row[col_index] == "NA":
                for row2 in self.data:
                    if row2[col_index] != "NA":
                        if row2[key_index] == row[key_index]:
                            sum += row2[col_index]
                            count += 1
                avg = sum/count
                avg = float(round(avg))
                row[col_index] = avg           

        return self


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
    
        self.remove_rows_with_missing_values()
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        summary_stats = []

        for attribute in col_names:
            stat_lst = []
            stat_lst.append(attribute)
            column = self.get_column(attribute)
            if len(column) == 0:
                summary_table = MyPyTable(header, summary_stats)
                return summary_table
            # get min, max, and mid
            minimum = min(column)
            maximum = max(column)
            mid = (minimum + maximum) / 2
            stat_lst.append(minimum) 
            stat_lst.append(maximum)
            stat_lst.append(mid)

            # get avg
            sum = 0
            count = 0
            for item in column:
                sum += item
                count += 1
            avg = sum/count
            stat_lst.append(avg)

            # get median
            column.sort()
            if len(column)%2 == 0:
                mid_high = int(len(column)/2)
                mid_low = mid_high - 1
                median = (column[mid_high] + column[mid_low])/2

            else:
                mid_index = int(len(column)/2)
                median = column[mid_index]

            stat_lst.append(median)

            summary_stats.append(stat_lst)

        summary_table = MyPyTable(header, summary_stats)

        return summary_table # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        is_match = 1
        any_match = 0
        header = []
        # get header for joined table
        for i in self.column_names:
            header.append(i)
        is_in = 0
        for j in other_table.column_names:
            is_in = 0
            for k in header:
                if j == k:
                    is_in = 1
            if is_in == 0:
                header.append(j)

        # join tables by iterating through rows 
        for row in self.data:
            any_match = 0
            for other_row in other_table.data:
                is_match = 1
                # check for matching key
                for name in key_column_names:
                    if row[self.column_names.index(name)] == other_row[other_table.column_names.index(name)]:
                        any_match = 1
                    else:
                        is_match = 0
                # if it is a key match, join the rows
                if is_match == 1: # if it is a match
                    joined_row = [] 
                    for column in row:
                        joined_row.append(column)
                    for i in range(0, len(other_table.column_names)):
                        same = 0
                        for j in range(0, len(self.column_names)):
                            if other_table.column_names[i] == self.column_names[j]:
                                same = 1
                        if same == 0:
                            joined_row.append(other_row[i])
            # append the joined row to the joined table
            if any_match == 1:
                joined_table.append(joined_row)

        # create a MyPyTable with the joined table data and the header
        inner_join = MyPyTable(header, joined_table)
        # remove duplicate from the end of the table if it exists
        duplicates = inner_join.find_duplicates(key_column_names)
        inner_join.drop_rows(duplicates)
        for item in duplicates:
            inner_join.data.append(item)
        
        # return the table
        return inner_join

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        
        # do an inner join of the tables
        inner_joined_table = self.perform_inner_join(other_table, key_column_names)
        # get header
        header = inner_joined_table.column_names
        more_rows = []
        # add rows not in inner join to table from self
        for row in self.data:
            for inner_row in inner_joined_table.data:
                # check if key matches
                key_match = 0
                for name in key_column_names:
                    if row[self.column_names.index(name)] == inner_row[inner_joined_table.column_names.index(name)]:
                        key_match += 1
                # if key matches, break from key for loop
                if key_match == len(key_column_names):
                    break
            # if row needs to be added, create row and add 
            if key_match != len(key_column_names):
                new_row = []
                for category in header:
                    inserted = 0
                    for name in self.column_names:
                        if category == name:
                            new_row.append(row[self.column_names.index(name)])
                            inserted = 1
                    if inserted == 0:
                        new_row.append("NA")
                more_rows.append(new_row)
    
        # add rows not in inner join to table from other table
        for row in other_table.data:
            for inner_row in inner_joined_table.data:
                # check if key matches
                key_match = 0
                for name in key_column_names:
                    if row[other_table.column_names.index(name)] == inner_row[inner_joined_table.column_names.index(name)]:
                        key_match += 1
                # if key matches, break from key for loop
                if key_match == len(key_column_names):
                    break
            # if row needs to be added, create row and add 
            if key_match != len(key_column_names):
                new_row = []
                for category in header:
                    inserted = 0
                    for name in other_table.column_names:
                        if category == name:
                            new_row.append(row[other_table.column_names.index(name)])
                            inserted = 1
                    if inserted == 0:
                        new_row.append("NA")
                more_rows.append(new_row)
        
        data = copy.deepcopy(inner_joined_table.data)
        for row in more_rows:
            data.append(row)
     

        outer_join = MyPyTable(header, data)

        return outer_join 